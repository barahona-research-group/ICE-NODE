from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from lib.ehr import CodingScheme
from lib.ehr.coding_scheme import NumericScheme
from lib.ehr.tvx_concepts import SegmentedAdmission, InpatientObservables
from lib.ml.embeddings import InICENODEEmbeddingsConfig, InterventionsEmbeddingsConfig, EmbeddedAdmission
from lib.ml.in_models import InICENODE, LeadPredictorName, InpatientModelConfig, AdmissionTrajectoryPrediction, \
    InterventionUncertaintyWeightingScheme
from lib.ml.model import Precomputes
from lib.utils import tree_hasnan


@pytest.fixture
def interventions_embeddings_config() -> InterventionsEmbeddingsConfig:
    return InterventionsEmbeddingsConfig(icu_inputs=10, icu_procedures=10, hosp_procedures=10, interventions=20)


@pytest.fixture
def inicenode_model_embeddings_config(
        interventions_embeddings_config: InterventionsEmbeddingsConfig) -> InICENODEEmbeddingsConfig:
    return InICENODEEmbeddingsConfig(dx_codes=50, demographic=None,
                                     interventions=interventions_embeddings_config)


@pytest.fixture(params=['monotonic', 'mlp'])
def lead_predictor_literal(request) -> LeadPredictorName:
    return request.param


@pytest.fixture
def inicenode_model_config(lead_predictor_literal: LeadPredictorName) -> InpatientModelConfig:
    return InpatientModelConfig(state=50, lead_predictor=lead_predictor_literal)


@pytest.fixture
def inicenode_model(inicenode_model_config: InpatientModelConfig,
                    inicenode_model_embeddings_config: InICENODEEmbeddingsConfig,
                    dx_scheme: CodingScheme,
                    observation_scheme: NumericScheme,
                    icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                    icu_inputs_grouping_data) -> InICENODE:
    return InICENODE(config=inicenode_model_config, embeddings_config=inicenode_model_embeddings_config,
                     lead_times=(1.,),
                     dx_codes_size=len(dx_scheme),
                     outcome_size=None,
                     icu_inputs_grouping=icu_inputs_grouping_data,
                     icu_procedures_size=len(icu_proc_scheme),
                     hosp_procedures_size=len(hosp_proc_scheme),
                     demographic_size=None,
                     observables_size=len(observation_scheme),
                     key=jrandom.PRNGKey(0))


@pytest.fixture
def embedded_admission(inicenode_model: InICENODE, segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return inicenode_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def inicenode_predictions(inicenode_model: InICENODE, segmented_admission: SegmentedAdmission,
                          embedded_admission: EmbeddedAdmission):
    return inicenode_model(admission=segmented_admission, embedded_admission=embedded_admission,
                           precomputes=Precomputes())


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_inicenode_predictions(inicenode_predictions):
    assert isinstance(inicenode_predictions, AdmissionTrajectoryPrediction)
    assert inicenode_predictions.leading_observable.value.shape == inicenode_predictions.admission.leading_observable.value.shape


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_model_grad_apply(inicenode_model: InICENODE, segmented_admission: SegmentedAdmission,
                          embedded_admission: EmbeddedAdmission):
    if len(segmented_admission.leading_observable) == 0:
        pytest.skip("No leading observable")

    def forward(model: InICENODE) -> jnp.ndarray:
        p = model(admission=segmented_admission, embedded_admission=embedded_admission, precomputes=Precomputes())
        return jnp.mean(p.leading_observable.value.mean())

    grad = eqx.filter_grad(forward)(inicenode_model)
    assert not tree_hasnan(grad)


@pytest.fixture
def intervention_uncertainty_weighting() -> InterventionUncertaintyWeightingScheme:
    return InterventionUncertaintyWeightingScheme(lead_times=(1.,), leading_observable_index=0)


@pytest.mark.usefixtures('jax_cpu_execution')
@pytest.fixture
def intervention_uncertainty_weighting_out(inicenode_model: InICENODE,
                                           inicenode_predictions: AdmissionTrajectoryPrediction,
                                           intervention_uncertainty_weighting: InterventionUncertaintyWeightingScheme,
                                           segmented_admission: SegmentedAdmission,
                                           embedded_admission: EmbeddedAdmission) -> Tuple[
    Tuple[float, int], InpatientObservables]:
    init_state = inicenode_predictions.trajectory.adjusted_state
    return intervention_uncertainty_weighting(f_obs_decoder=inicenode_model.f_obs_dec,
                                              f_ode_dyn=inicenode_model.f_dyn,
                                              initial_states=init_state,
                                              admission=segmented_admission,
                                              embedded_admission=embedded_admission,
                                              precomputes=Precomputes())

@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_intervention_uncertainty_weighting_out(intervention_uncertainty_weighting_out):
    assert len(intervention_uncertainty_weighting_out) == 2