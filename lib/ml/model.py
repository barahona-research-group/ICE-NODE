"""Abstract class for predictive EHR models."""

from __future__ import annotations
from typing import (List, TYPE_CHECKING, Callable, Union, Tuple, Optional, Any,
                    Dict, Type)
from abc import abstractmethod, ABCMeta
import zipfile
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ..utils import tqdm_constructor, translate_path
from ..ehr import (Patients, Patient, DemographicVectorConfig, DatasetScheme,
                   Predictions, Admission, TrajectoryConfig, PatientTrajectory,
                   AdmissionVisualisables, InpatientInput,
                   InpatientObservables, AdmissionPrediction)
from ..base import Config, Module, Data

from .embeddings import (PatientEmbedding, PatientEmbeddingConfig,
                         EmbeddedAdmission)

if TYPE_CHECKING:
    import optuna


class ModelConfig(Config):
    emb: PatientEmbeddingConfig = PatientEmbeddingConfig()


class ModelRegularisation(Config):
    L_l1: float = 0.0
    L_l2: float = 0.0


class Precomputes(Data):
    pass


class AbstractModel(Module):
    _f_emb: PatientEmbedding
    _f_dx_dec: Callable

    config: ModelConfig = eqx.static_field()

    @property
    @abstractmethod
    def dyn_params_list(self):
        pass

    @classmethod
    def external_argnames(cls):
        return []

    def params_list(self, pytree: Optional[Any] = None):
        if pytree is None:
            pytree = self
        return jtu.tree_leaves(eqx.filter(pytree, eqx.is_inexact_array))

    def params_list_mask(self, pytree):
        assert isinstance(pytree, dict) and 'dyn' in pytree and len(
            pytree) == 2, 'Expected a "dyn" label in Dict'
        (other_key, ) = set(pytree.keys()) - {'dyn'}

        params_list = self.params_list()
        dyn_params = self.dyn_params_list
        is_dyn_param = lambda x: any(x is y for y in dyn_params)
        mask = {'dyn': [is_dyn_param(x) for x in params_list]}
        mask[other_key] = [not m for m in mask['dyn']]
        return mask

    @classmethod
    def _assert_demo_dim(cls, config: ModelConfig, scheme: DatasetScheme,
                         demographic_vector_config: DemographicVectorConfig):
        demo_vec_dim = scheme.demographic_vector_size(
            demographic_vector_config)
        assert ((demo_vec_dim == 0 and config.emb.demo == 0) or
                (demo_vec_dim > 0 and config.emb.demo > 0)), \
            f"Model dimensionality for demographic embedding size "\
            f"({config.emb.demo}) and input demographic vector size "\
            f"({demo_vec_dim}) must both be zero or non-zero."

    def precomputes(self, inpatients: Patients):
        return Precomputes()

    @abstractmethod
    def __call__(self,
                 x: Union[Patient, Admission],
                 embedded_x: Union[List[EmbeddedAdmission], EmbeddedAdmission],
                 precomputes: Precomputes,
                 regularisation: Optional[ModelRegularisation] = None,
                 store_embeddings: Optional[TrajectoryConfig] = None):
        pass

    @abstractmethod
    def batch_predict(
            self,
            patients: Patients,
            leave_pbar: bool = False,
            regularisation: Optional[ModelRegularisation] = None,
            store_embeddings: Optional[TrajectoryConfig] = None
    ) -> Predictions:
        pass

    @property
    @abstractmethod
    def counts_ignore_first_admission(self):
        pass

    # @classmethod
    # def sample_reg_regularisation(cls, trial: optuna.Trial):
    #     return {
    #         'L_l1': 0,  #trial.suggest_float('l1', 1e-8, 5e-3, log=True),
    #         'L_l2': 0  # trial.suggest_float('l2', 1e-8, 5e-3, log=True),
    #     }

    @classmethod
    def sample_model_config(cls, trial: optuna.Trial):
        return {}

    def weights(self):

        def _weights(x):
            w = []
            if not hasattr(x, '__dict__'):
                return w
            for k, v in x.__dict__.items():
                if 'weight' in k:
                    w.append(v)
            return tuple(w)

        has_weight = lambda leaf: len(_weights(leaf)) > 0
        # Valid for eqx.nn.MLP and ml.base_models.GRUDynamics
        return sum((_weights(x)
                    for x in jtu.tree_leaves(self, is_leaf=has_weight)
                    if has_weight(x)), ())

    def l1(self):
        return sum(jnp.abs(w).sum() for w in jtu.tree_leaves(self.weights()))

    def l2(self):
        return sum(
            jnp.square(w).sum() for w in jtu.tree_leaves(self.weights()))

    def load_params(self, params_file):
        """
        Load the parameters in `params_file`\
            filepath and return as PyTree Object.
        """
        with open(translate_path(params_file), 'rb') as file_rsc:
            return eqx.tree_deserialise_leaves(file_rsc, self)

    def write_params(self, params_file):
        """
        Store the parameters (PyTree object) into a new file
        given by `params_file`.
        """
        with open(translate_path(params_file), 'wb') as file_rsc:
            eqx.tree_serialise_leaves(file_rsc, self)

    def load_params_from_archive(self, zipfile_fname: str, params_fname: str):

        with zipfile.ZipFile(translate_path(zipfile_fname),
                             compression=zipfile.ZIP_STORED,
                             mode="r") as archive:
            with archive.open(params_fname, "r") as zip_member:
                return eqx.tree_deserialise_leaves(zip_member, self)

    def pathwise_params(self):

        def label(x):
            if hasattr(x, 'key'):
                return x.key
            if hasattr(x, 'name'):
                return x.name
            if hasattr(x, 'idx'):
                return str(x.idx)
            return 'unknown'

        params_part, _ = eqx.partition(self, eqx.is_inexact_array)
        params_w_path, _ = jtu.tree_flatten_with_path(params_part)
        paths, params = zip(*params_w_path)
        dotted_paths = map(lambda path: '.'.join(map(label, path)), paths)
        return dict(zip(dotted_paths, params))

    @eqx.filter_jit
    def pathwise_params_stats(self):
        params = self.pathwise_params()
        return {
            k: {
                'mean': jnp.nanmean(v),
                'std': jnp.nanstd(v),
                'min': jnp.nanmin(v),
                'max': jnp.nanmax(v),
                'l1': jnp.abs(v).sum(),
                'l2': jnp.square(v).sum(),
                'nans': jnp.isnan(v).sum(),
                'size': jnp.size(v)
            }
            for k, v in params.items()
        }


class InpatientModel(AbstractModel):

    @property
    def counts_ignore_first_admission(self):
        return False

    @classmethod
    def external_argnames(cls):
        return [
            "schemes", "demographic_vector_config",
            "leading_observable_config", "key"
        ]

    def decode_lead_trajectory(
            self, trajectory: PatientTrajectory) -> PatientTrajectory:
        raise NotImplementedError

    def decode_obs_trajectory(
            self, trajectories: PatientTrajectory) -> PatientTrajectory:
        raise NotImplementedError

    def leading_observable_trajectory(self, predictions: Predictions):
        dic = {}
        for sid, spreds in predictions.items():
            dic[sid] = {}
            for aid, apreds in spreds.items():
                assert apreds.trajectory is not None
                trajectory = apreds.trajectory
                lead_trajectory = self.decode_lead_trajectory(trajectory)
                dic[sid][aid] = lead_trajectory.to_cpu()
        return dic

    def observables_trajectory(self, predictions: Predictions):
        dic = {}
        for sid, spreds in predictions.items():
            dic[sid] = {}
            for aid, apreds in spreds.items():
                assert apreds.trajectory is not None
                trajectory = apreds.trajectory
                obs_trajectory = self.decode_obs_trajectory(trajectory)
                dic[sid][aid] = obs_trajectory.to_cpu()
        return dic

    def predict_visualisables(self,
                              inpatients: Patients,
                              store_embeddings: TrajectoryConfig,
                              leave_pbar: bool = False):
        assert store_embeddings is not None
        predictions = self.batch_predict(inpatients,
                                         leave_pbar=leave_pbar,
                                         regularisation=None,
                                         store_embeddings=store_embeddings)
        lead_trajectories = self.leading_observable_trajectory(predictions)
        obs_trajectories = self.observables_trajectory(predictions)

        predictions = predictions.to_cpu().defragment_observables()

        scaled_visualisables = {}
        unscaled_visualisables = {}
        for sid, spreds in predictions.items():
            scaled_visualisables[sid] = {}
            unscaled_visualisables[sid] = {}
            for aid, apreds in spreds.items():
                obs = apreds.admission.observables
                state_adjustment_time = obs.time
                vis = AdmissionVisualisables(
                    obs_pred_trajectory=obs_trajectories[sid][aid],
                    lead_pred_trajectory=lead_trajectories[sid][aid],
                    obs=apreds.admission.observables,
                    lead=apreds.admission.leading_observable,
                    interventions=apreds.admission.interventions,
                    state_adjustment_time=state_adjustment_time)
                vis_unscaled = vis.unscaled(inpatients)

                vis = vis.groupby_code(inpatients)
                vis = vis.listify_interventions(inpatients)
                vis_unscaled = vis_unscaled.groupby_code(inpatients)
                vis_unscaled = vis_unscaled.listify_interventions(inpatients)

                scaled_visualisables[sid][aid] = vis
                unscaled_visualisables[sid][aid] = vis_unscaled

        return scaled_visualisables, unscaled_visualisables

    def batch_predict(
            self,
            inpatients: Patients,
            leave_pbar: bool = False,
            regularisation: Optional[ModelRegularisation] = None,
            store_embeddings: Optional[TrajectoryConfig] = None
    ) -> Predictions:
        total_int_days = inpatients.interval_days()
        precomputes = self.precomputes(inpatients)
        inpatients_emb = {
            i: self._f_emb(subject)
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar)
        }

        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i+1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for adm, adm_e in zip(inpatient.admissions,
                                      embedded_admissions):
                    results.add(subject_id=subject_id,
                                prediction=self(
                                    adm,
                                    adm_e,
                                    regularisation=regularisation,
                                    store_embeddings=store_embeddings,
                                    precomputes=precomputes))
                    pbar.update(adm.interval_days)
            return results.filter_nans()


class OutpatientModel(AbstractModel):

    @property
    def counts_ignore_first_admission(self):
        return True

    @classmethod
    def external_argnames(cls):
        return ["schemes", "demographic_vector_config", "key"]

    def batch_predict(
            self,
            inpatients: Patients,
            leave_pbar: bool = False,
            regularisation: Optional[ModelRegularisation] = None,
            store_embeddings: Optional[TrajectoryConfig] = None
    ) -> Predictions:
        total_int_days = inpatients.d2d_interval_days()
        precomputes = self.precomputes(inpatients)
        inpatients_emb = {
            i: self._f_emb(subject)
            for i, subject in tqdm_constructor(inpatients.subjects.items(),
                                               desc="Embedding",
                                               unit='subject',
                                               leave=leave_pbar)
        }
        r_bar = '| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
        bar_format = '{l_bar}{bar}' + r_bar
        with tqdm_constructor(total=total_int_days,
                              bar_format=bar_format,
                              unit='longitudinal-days',
                              leave=leave_pbar) as pbar:
            results = Predictions()
            for i, subject_id in enumerate(inpatients.subjects.keys()):
                pbar.set_description(
                    f"Subject: {subject_id} ({i+1}/{len(inpatients)})")
                inpatient = inpatients.subjects[subject_id]
                embedded_admissions = inpatients_emb[subject_id]
                for pred in self(inpatient,
                                 embedded_admissions,
                                 precomputes=precomputes,
                                 regularisation=regularisation,
                                 store_embeddings=store_embeddings):
                    results.add(subject_id=subject_id, prediction=pred)
                pbar.update(inpatient.d2d_interval_days)
            return results.filter_nans()

    def patient_embeddings(self, patients: Patients):
        pass


#     @classmethod
#     def from_config(cls, conf: Dict[str, Any], patients: Patients,
#                     train_split: List[int], key: "jax.random.PRNGKey"):
#         decoder_input_size = cls.decoder_input_size(conf)
#         emb_models = embeddings_from_conf(conf["emb"], patients, train_split,
#                                           decoder_input_size)
#         control_size = patients.control_dim
#         return cls(**emb_models,
#                    **conf["model"],
#                    outcome=patients.outcome_extractor,
#                    control_size=control_size,
#                    key=key)
