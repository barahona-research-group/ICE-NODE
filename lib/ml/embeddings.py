from __future__ import annotations
from typing import (Any, Dict, List, Callable, Optional, Tuple)
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import numpy as np

from ..ehr import (Patient, Admission, StaticInfo, DatasetScheme,
                   InpatientInput, InpatientInterventions, DemographicVectorConfig, CodingScheme)
from ..base import Config, VxData


class MaskedAggregator(eqx.Module):
    """
    A class representing a masked aggregator.

    Attributes:
        mask (jnp.array): The mask used for aggregation.
    """

    mask: jnp.array = eqx.static_field()

    def __init__(self,
                 subsets: jax.Array | List[int],
                 input_size: int):
        super().__init__()
        mask = np.zeros((len(subsets), input_size), dtype=bool)
        for i, s in enumerate(subsets):
            mask[i, s] = True
        self.mask = jnp.array(mask)

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


class MaskedPerceptron(MaskedAggregator):
    """
    A masked perceptron model for classification tasks.

    Parameters:
    - subsets: an array of subsets used for masking.
    - input_size: the size of the input features.
    - key: a PRNGKey for random initialization.
    - backend: the backend framework to use (default: 'jax').

    Attributes:
    - linear: a linear layer for the perceptron.

    Methods:
    - __call__(self, x): performs forward pass of the perceptron.
    """

    linear: eqx.nn.Linear

    def __init__(self,
                 subsets: jax.Array,
                 input_size: int,
                 key: "jax.random.PRNGKey"):
        """
        Initialize the MaskedPerceptron class.

        Args:
            subsets (Array): the subsets of input features.
            input_size (int): the size of the input.
            key (jax.random.PRNGKey): the random key for initialization.
        """
        super().__init__(subsets, input_size)
        self.linear = eqx.nn.Linear(input_size,
                                    len(subsets),
                                    use_bias=False,
                                    key=key)

    @property
    def weight(self):
        """
        Returns the weight of the linear layer.

        Returns:
            Array: the weights of the linear layer.
        """
        return self.linear.weight

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Performs forward pass of the perceptron.

        Args:
            x (Array): the input features.

        Returns:
            Array: the output of the perceptron.
        """
        return (self.weight * self.mask) @ x


class MaskedSum(MaskedAggregator):

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        performs a masked sum aggregation on the input array x.

        Args:
            x (Array): input array to aggregate.

        Returns:
            Array: sum of x for True mask locations.
        """
        return self.mask @ x


class MaskedOr(MaskedAggregator):

    def __call__(self, x):
        """Performs a masked OR aggregation.

        For each row of the input array `x`, performs a logical OR operation between
        elements of that row and the mask. Returns a boolean array indicating if there
        was at least one `True` value for each row.

        Args:
            x: Input array to aggregate. Can be numpy ndarray or jax ndarray.

        Returns:
            Boolean ndarray indicating if there was at least one True value in each
            row of x after applying the mask.
        """
        return jnp.any(self.mask & (x != 0), axis=1)


class AggregateRepresentation(eqx.Module):
    """
    AggregateRepresentation aggregates input codes into target codes.

    It initializes masked aggregators based on the target scheme's
    aggregation and aggregation groups. On call, it splits the input,
    aggregates each split with the corresponding aggregator, and
    concatenates the results.

    Handles both jax and numpy arrays for the input.

    Attributes:
        aggregators: a list of masked aggregators.
        splits: a tuple of integers indicating the splits of the input.

    Methods:
        __call__(self, x): performs the aggregation.
    """
    aggregators: List[MaskedAggregator]
    splits: Tuple[int] = eqx.static_field()

    def __init__(self,
                 source_scheme: CodingScheme,
                 target_scheme: CodingScheme,
                 key: "jax.random.PRNGKey" = None):
        """
        Initializes an AggregateRepresentation.

        Constructs the masked aggregators based on the target scheme's
        aggregation and aggregation groups. Splits the input into sections
        for each aggregator.

        Args:
            source_scheme: Source coding scheme to aggregate from
            target_scheme: Target coding scheme to aggregate to
            key: JAX PRNGKey for initializing perceptrons
        """
        super().__init__()
        self.aggregators = []
        aggs = target_scheme.aggregation
        agg_grps = target_scheme.aggregation_groups
        grps = target_scheme.groups
        splits = []

        def is_contagious(x):
            return x.max() - x.min() == len(x) - 1 and len(set(x)) == len(x)

        for agg in aggs:
            selectors = []
            agg_offset = len(source_scheme)
            for grp in agg_grps[agg]:
                input_codes = grps[grp]
                input_index = sorted(source_scheme.index[c]
                                     for c in input_codes)
                input_index = jnp.array(input_index, dtype=int)
                assert is_contagious(input_index), (
                    f"Selectors must be contiguous, but {input_index} is not. Codes: {input_codes}. Group: {grp}"
                )
                agg_offset = min(input_index.min().item(), agg_offset)
                selectors.append(input_index)
            selectors = [s - agg_offset for s in selectors]
            agg_input_size = sum(len(s) for s in selectors)
            max_index = max(s.max().item() for s in selectors)
            assert max_index == agg_input_size - 1, (
                f"Selectors must be contiguous, max index is {max_index} but size is {agg_input_size}"
            )
            splits.append(agg_input_size)

            if agg == 'w_sum':
                self.aggregators.append(
                    MaskedPerceptron(selectors, agg_input_size, key))
                (key,) = jrandom.split(key, 1)
            elif agg == 'sum':
                self.aggregators.append(
                    MaskedSum(selectors, agg_input_size))
            elif agg == 'or':
                self.aggregators.append(
                    MaskedOr(selectors, agg_input_size))
            else:
                raise ValueError(f"Aggregation {agg} not supported")
        splits = jnp.cumsum([0] + splits)[1:-1]
        self.splits = tuple(splits.tolist())

    @eqx.filter_jit
    def __call__(self, inpatient_input: jax.Array) -> jax.Array:
        """
        Apply aggregators to the input data.

        Args:
            inpatient_input (Array): the input data to be processed.

        Returns:
            Array: the processed data after applying aggregators.
        """
        if isinstance(inpatient_input, jax.Array):
            splitted = jnp.hsplit(inpatient_input, self.splits)
            return jnp.hstack(
                [agg(x) for x, agg in zip(splitted, self.aggregators)])

        splitted = jnp.hsplit(inpatient_input, self.splits)
        return jnp.hstack(
            [agg(x) for x, agg in zip(splitted, self.aggregators)])


class EmbeddedAdmission(VxData):
    pass


class EmbeddedInAdmission(EmbeddedAdmission):
    dx0: jnp.ndarray
    inp_proc_demo: Optional[jnp.ndarray]


class EmbeddedOutAdmission(EmbeddedAdmission):
    dx: jnp.ndarray
    demo: Optional[jnp.ndarray]


class LiteEmbeddedInAdmission(EmbeddedAdmission):
    dx0: jnp.ndarray
    demo: Optional[jnp.ndarray]


class PatientEmbeddingConfig(Config):
    dx: int = 30
    demo: int = 5


class OutpatientEmbeddingConfig(PatientEmbeddingConfig):
    pass


class InpatientLiteEmbeddingConfig(PatientEmbeddingConfig):
    pass


class InpatientEmbeddingConfig(PatientEmbeddingConfig):
    inp: int = 10
    proc: int = 10
    inp_proc_demo: int = 15


class PatientEmbedding(eqx.Module):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    _f_dx_emb: Callable
    _f_dem_emb: Callable

    def __init__(self, config: OutpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        super().__init__()
        (dx_emb_key, dem_emb_key) = jrandom.split(key, 2)

        self._f_dx_emb = eqx.nn.MLP(len(schemes[1].dx_discharge),
                                    config.dx,
                                    width_size=config.dx * 5,
                                    depth=1,
                                    final_activation=jnp.tanh,
                                    key=dx_emb_key)

        demo_input_size = schemes[1].demographic_vector_size(
            demographic_vector_config)
        if demo_input_size > 0:
            self._f_dem_emb = eqx.nn.MLP(demo_input_size,
                                         config.demo,
                                         config.demo * 5,
                                         depth=1,
                                         final_activation=jnp.tanh,
                                         key=dem_emb_key)
        else:
            self._f_dem_emb = lambda x: jnp.array([], dtype=jnp.float16)

    @eqx.filter_jit
    def dx_embdeddings(self):
        in_size = self._f_dx_emb.in_size
        return eqx.filter_vmap(self._f_dx_emb)(jnp.eye(in_size))

    @abstractmethod
    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedAdmission:
        """
        Embeds an admission into fixed vectors as described above.
        """
        pass

    def __call__(self, inpatient: Patient) -> List[EmbeddedAdmission]:
        """
        Embeds all the admissions of an inpatient into fixed vectors as \
        described above.
        """
        return [
            self.embed_admission(inpatient.static_info, admission)
            for admission in inpatient.admissions
        ]


class OutpatientEmbedding(PatientEmbedding):

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray,
                         dx_vec: jnp.ndarray) -> EmbeddedOutAdmission:
        dx_emb = self._f_dx_emb(dx_vec)
        demo_e = self._f_dem_emb(demo)
        return EmbeddedOutAdmission(dx=dx_emb, demo=demo_e)

    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedOutAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes.vec)


class InpatientLiteEmbedding(PatientEmbedding):

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray,
                         dx_vec: jnp.ndarray) -> EmbeddedOutAdmission:
        dx_emb = self._f_dx_emb(dx_vec)
        demo_e = self._f_dem_emb(demo)
        return LiteEmbeddedInAdmission(dx0=dx_emb, demo=demo_e)

    def embed_admission(self, static_info: StaticInfo,
                        admission: Admission) -> EmbeddedOutAdmission:
        """ Embeds an admission into fixed vectors as described above."""
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes_history.vec)


class InpatientEmbedding(PatientEmbedding):
    """
    Embeds an inpatient admission into fixed vectors:
        - Embdedded discharge codes history.
        - A sequence of embedded vectors each fusing the input, procedure \
            and demographic information.
    """
    _f_inp_agg: Callable
    _f_inp_emb: Callable
    _f_proc_emb: Callable
    _f_int_emb: Callable

    def __init__(self, config: InpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):
        (super_key, inp_agg_key, inp_emb_key, proc_emb_key,
         int_emb_key) = jrandom.split(key, 5)

        if schemes[1].demographic_vector_size(demographic_vector_config) == 0:
            config = eqx.tree_at(lambda d: d.demo, config, 0)

        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         key=super_key)
        self._f_inp_agg = AggregateRepresentation(schemes[0].int_input,
                                                  schemes[1].int_input,
                                                  inp_agg_key, 'jax')
        self._f_inp_emb = eqx.nn.MLP(len(schemes[1].int_input),
                                     config.inp,
                                     config.inp * 5,
                                     final_activation=jnp.tanh,
                                     depth=1,
                                     key=inp_emb_key)
        self._f_proc_emb = eqx.nn.MLP(len(schemes[1].int_proc),
                                      config.proc,
                                      config.proc * 5,
                                      final_activation=jnp.tanh,
                                      depth=1,
                                      key=proc_emb_key)
        self._f_int_emb = eqx.nn.MLP(config.inp + config.proc + config.demo,
                                     config.inp_proc_demo,
                                     config.inp_proc_demo * 5,
                                     final_activation=jnp.tanh,
                                     depth=1,
                                     key=int_emb_key)

    @eqx.filter_jit
    def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
        """Embeds the demographics into a fixed vector."""
        return self._f_dem_emb(demo)

    @eqx.filter_jit
    def _embed_segment(self, inp: InpatientInput, proc: InpatientInput,
                       demo_e: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds a  of the intervention (procedures and inputs) \
        and demographics into a fixed vector.
        """

        inp_emb = self._f_inp_emb(self._f_inp_agg(inp))
        proc_emb = self._f_proc_emb(proc)
        return self._f_int_emb(jnp.hstack([inp_emb, proc_emb, demo_e]))

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self._f_dx_emb(x)

    @eqx.filter_jit
    def _embed_admission(self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
                         segmented_inp: jnp.ndarray,
                         segmented_proc: jnp.ndarray) -> EmbeddedInAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        demo_e = self._embed_demo(demo)
        dx_emb = self.embed_dx(dx_history_vec)

        def _embed_segment(inp, proc):
            return self._embed_segment(inp, proc, demo_e)

        inp_proc_demo = jax.vmap(_embed_segment)(segmented_inp, segmented_proc)
        return EmbeddedInAdmission(dx0=dx_emb, inp_proc_demo=inp_proc_demo)

    def embed_admission(self, static_info: StaticInfo, admission: Admission):
        demo = static_info.demographic_vector(admission.admission_dates[0])
        return self._embed_admission(demo, admission.dx_codes_history.vec,
                                     admission.interventions.segmented_input,
                                     admission.interventions.segmented_proc)


class DeepMindPatientEmbeddingConfig(Config):
    dx: int = 30
    demo: int = 5
    obs: int = 20
    sequence: int = 50


class DeepMindEmbeddedAdmission(VxData):
    dx0: jnp.ndarray
    demo: jnp.ndarray
    sequence: jnp.ndarray


class DeepMindPatientEmbedding(PatientEmbedding):
    _f_dx_emb: Callable
    _f_dem_emb: Callable
    _f_obs_emb: Callable
    _f_sequence_emb: Callable

    def __init__(self, config: OutpatientEmbeddingConfig,
                 schemes: Tuple[DatasetScheme],
                 demographic_vector_config: DemographicVectorConfig,
                 key: "jax.random.PRNGKey"):

        (super_key, obs_emb_key, sequence_emb_key) = jrandom.split(key, 3)
        super().__init__(config=config,
                         schemes=schemes,
                         demographic_vector_config=demographic_vector_config,
                         key=super_key)

        self._f_obs_emb = eqx.nn.MLP(len(schemes[1].obs) * 2,
                                     config.obs,
                                     width_size=config.obs * 3,
                                     depth=1,
                                     final_activation=jnp.tanh,
                                     key=obs_emb_key)

        self._f_sequence_emb = eqx.nn.MLP(config.dx + config.demo + config.obs,
                                          config.sequence,
                                          config.sequence * 2,
                                          depth=1,
                                          final_activation=jnp.tanh,
                                          key=sequence_emb_key)

    @eqx.filter_jit
    def embed_dx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embeds the discharge codes history into a fixed vector."""
        return self._f_dx_emb(x)

    @eqx.filter_jit
    def dx_embdeddings(self):
        in_size = self._f_dx_emb.in_size
        return eqx.filter_vmap(self._f_dx_emb)(jnp.eye(in_size))

    @eqx.filter_jit
    def _embed_demo(self, demo: jnp.ndarray) -> jnp.ndarray:
        """Embeds the demographics into a fixed vector."""
        return self._f_dem_emb(demo)

    def _embed_admission(
            self, demo: jnp.ndarray, dx_history_vec: jnp.ndarray,
            observables: jnp.ndarray) -> DeepMindEmbeddedAdmission:
        """ Embeds an admission into fixed vectors as described above."""

        dx_emb = self.embed_dx(dx_history_vec)
        demo_e = self._embed_demo(demo)
        obs_e = jax.vmap(self._f_obs_emb)(observables)

        def _embed_sequence_features(obs_e_i):
            return self._f_sequence_emb(jnp.hstack((dx_emb, demo_e, obs_e_i)))

        sequence_e = [_embed_sequence_features(obs_e_i) for obs_e_i in obs_e]
        return DeepMindEmbeddedAdmission(dx0=dx_emb,
                                         sequence=sequence_e,
                                         demo=demo_e)

    def embed_admission(self, static_info: StaticInfo, admission: Admission):
        demo = static_info.demographic_vector(admission.admission_dates[0])

        assert (not isinstance(admission.observables,
                               list)), "Observables should not be fragmented"

        observables = jnp.hstack(
            (admission.observables.value, admission.observables.mask))

        return self._embed_admission(demo, admission.dx_codes_history.vec,
                                     observables)
