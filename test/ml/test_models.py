import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from lib.ehr import CodingScheme
from lib.ehr.coding_scheme import NumericScheme
from lib.ehr.tvx_concepts import SegmentedAdmission
from lib.metric.loss_wrap import ProbObsPredictionLoss, ImputedProbObsPredictionLoss
from lib.metric.metrics import ObsPredictionLoss
from lib.ml.artefacts import AdmissionsPrediction, AdmissionPrediction
from lib.ml.embeddings import InICENODEEmbeddingsConfig, InterventionsEmbeddingsConfig, EmbeddedAdmission
from lib.ml.exp_ode_icnn import AutoODEICNN, ODEICNNConfig
from lib.ml.in_models import InICENODE, LeadPredictorName, AdmissionTrajectoryPrediction, \
    DynamicsLiteral, ICENODEConfig, GRUODEBayes, AdmissionGRUODEBayesPrediction, InICENODELiteICNNImpute, \
    StochasticInICENODELite, StochasticMechanisticICENODE, InKoopman
from lib.ml.model import Precomputes
from lib.utils import tree_hasnan


@pytest.fixture
def interventions_embeddings_config() -> InterventionsEmbeddingsConfig:
    return InterventionsEmbeddingsConfig(icu_inputs=10, icu_procedures=10, hosp_procedures=10, interventions=20)


@pytest.fixture
def in_model_embeddings_config(
        interventions_embeddings_config: InterventionsEmbeddingsConfig) -> InICENODEEmbeddingsConfig:
    return InICENODEEmbeddingsConfig(dx_codes=50, demographic=None,
                                     interventions=interventions_embeddings_config)


@pytest.fixture(params=['monotonic', 'mlp'])
def lead_predictor_literal(request) -> LeadPredictorName:
    return request.param


@pytest.fixture(params=['mlp', 'gru'])
def inicenode_dynamics(request) -> DynamicsLiteral:
    return request.param


@pytest.fixture
def inicenode_model_config(lead_predictor_literal: LeadPredictorName,
                           inicenode_dynamics: DynamicsLiteral) -> ICENODEConfig:
    return ICENODEConfig(state=50, lead_predictor=lead_predictor_literal,
                         dynamics=inicenode_dynamics)


@pytest.fixture
def inicenode_model(inicenode_model_config: ICENODEConfig,
                    in_model_embeddings_config: InICENODEEmbeddingsConfig,
                    dx_scheme: CodingScheme,
                    observation_scheme: NumericScheme,
                    icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                    icu_inputs_grouping_data) -> InICENODE:
    return InICENODE(config=inicenode_model_config, embeddings_config=in_model_embeddings_config,
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
@pytest.mark.usefixtures('jax_cpu_execution')
def inicenode_predictions(inicenode_model: InICENODE, segmented_admission: SegmentedAdmission,
                          inicenode_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return inicenode_model(admission=segmented_admission, embedded_admission=inicenode_embedded_admission,
                           precomputes=Precomputes())


@pytest.fixture
def inicenode_embedded_admission(inicenode_model: InICENODE,
                                 segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return inicenode_model.f_emb(segmented_admission, None)


@pytest.fixture
def in_model_config() -> ICENODEConfig:
    return ICENODEConfig(state=50, lead_predictor='mlp',
                         dynamics='gru')


@pytest.fixture
def gru_ode_bayes_model(in_model_config: ICENODEConfig,
                        in_model_embeddings_config: InICENODEEmbeddingsConfig,
                        dx_scheme: CodingScheme,
                        observation_scheme: NumericScheme,
                        icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                        icu_inputs_grouping_data) -> GRUODEBayes:
    return GRUODEBayes(config=in_model_config, embeddings_config=in_model_embeddings_config,
                       lead_times=(1.,),
                       dx_codes_size=len(dx_scheme),
                       icu_inputs_grouping=icu_inputs_grouping_data,
                       icu_procedures_size=len(icu_proc_scheme),
                       hosp_procedures_size=len(hosp_proc_scheme),
                       demographic_size=None,
                       observables_size=len(observation_scheme),
                       key=jrandom.PRNGKey(0))


@pytest.fixture
def gru_ode_bayes_embedded_admission(gru_ode_bayes_model: InICENODE,
                                     segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return gru_ode_bayes_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def gru_ode_bayes_predictions(gru_ode_bayes_model: GRUODEBayes, segmented_admission: SegmentedAdmission,
                              gru_ode_bayes_embedded_admission: EmbeddedAdmission) -> AdmissionGRUODEBayesPrediction:
    return gru_ode_bayes_model(admission=segmented_admission, embedded_admission=gru_ode_bayes_embedded_admission,
                               precomputes=Precomputes())


@pytest.fixture
def incenode_icnn_model(in_model_config: ICENODEConfig,
                        in_model_embeddings_config: InICENODEEmbeddingsConfig,
                        dx_scheme: CodingScheme,
                        observation_scheme: NumericScheme,
                        icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                        icu_inputs_grouping_data) -> InICENODELiteICNNImpute:
    return InICENODELiteICNNImpute(config=in_model_config, embeddings_config=in_model_embeddings_config,
                                   lead_times=(1.,),
                                   dx_codes_size=len(dx_scheme),
                                   icu_inputs_grouping=icu_inputs_grouping_data,
                                   icu_procedures_size=len(icu_proc_scheme),
                                   hosp_procedures_size=len(hosp_proc_scheme),
                                   demographic_size=None,
                                   observables_size=len(observation_scheme),
                                   key=jrandom.PRNGKey(0))


@pytest.fixture
def icenode_icnn_embedded_admission(incenode_icnn_model: InICENODELiteICNNImpute,
                                    segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return incenode_icnn_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def icenode_icnn_predictions(incenode_icnn_model: InICENODELiteICNNImpute, segmented_admission: SegmentedAdmission,
                             icenode_icnn_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return incenode_icnn_model(admission=segmented_admission, embedded_admission=icenode_icnn_embedded_admission,
                               precomputes=Precomputes())


@pytest.fixture
def auto_icnn_model(in_model_config: ODEICNNConfig,
                    in_model_embeddings_config: InICENODEEmbeddingsConfig,
                    dx_scheme: CodingScheme,
                    observation_scheme: NumericScheme,
                    icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                    icu_inputs_grouping_data) -> AutoODEICNN:
    return AutoODEICNN(config=ODEICNNConfig(state=in_model_config.state,
                                            dynamics=in_model_config.dynamics,
                                            memory_ratio=0.5),
                       observables_size=len(observation_scheme), key=jrandom.PRNGKey(0))


@pytest.fixture
def auto_icnn_embedded_admission(auto_icnn_model: AutoODEICNN,
                                 segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return auto_icnn_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def auto_icnn_predictions(auto_icnn_model: AutoODEICNN, segmented_admission: SegmentedAdmission,
                          auto_icnn_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return auto_icnn_model(admission=segmented_admission, embedded_admission=auto_icnn_embedded_admission,
                           precomputes=Precomputes())


@pytest.fixture
def stochastic_icenode_model(in_model_config: ICENODEConfig,
                             in_model_embeddings_config: InICENODEEmbeddingsConfig,
                             dx_scheme: CodingScheme,
                             observation_scheme: NumericScheme,
                             icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                             icu_inputs_grouping_data) -> StochasticInICENODELite:
    return StochasticInICENODELite(config=in_model_config, embeddings_config=in_model_embeddings_config,
                                   lead_times=(1.,),
                                   dx_codes_size=len(dx_scheme),
                                   icu_inputs_grouping=icu_inputs_grouping_data,
                                   icu_procedures_size=len(icu_proc_scheme),
                                   hosp_procedures_size=len(hosp_proc_scheme),
                                   demographic_size=None,
                                   observables_size=len(observation_scheme),
                                   key=jrandom.PRNGKey(0))


@pytest.fixture
def stochastic_icenode_embedded_admission(stochastic_icenode_model: StochasticInICENODELite,
                                          segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return stochastic_icenode_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def stochastic_icenode_predictions(stochastic_icenode_model: StochasticInICENODELite,
                                   segmented_admission: SegmentedAdmission,
                                   stochastic_icenode_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return stochastic_icenode_model(admission=segmented_admission,
                                    embedded_admission=stochastic_icenode_embedded_admission,
                                    precomputes=Precomputes())


@pytest.fixture
def stochastic_mechanistic_icenode_model(in_model_config: ICENODEConfig,
                                         in_model_embeddings_config: InICENODEEmbeddingsConfig,
                                         dx_scheme: CodingScheme,
                                         observation_scheme: NumericScheme,
                                         icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                                         icu_inputs_grouping_data) -> StochasticMechanisticICENODE:
    return StochasticMechanisticICENODE(config=in_model_config, embeddings_config=in_model_embeddings_config,
                                        lead_times=(1.,),
                                        dx_codes_size=len(dx_scheme),
                                        icu_inputs_grouping=icu_inputs_grouping_data,
                                        icu_procedures_size=len(icu_proc_scheme),
                                        hosp_procedures_size=len(hosp_proc_scheme),
                                        demographic_size=None,
                                        observables_size=len(observation_scheme),
                                        key=jrandom.PRNGKey(0))


@pytest.fixture
def stochastic_mechanistic_icenode_embedded_admission(
        stochastic_mechanistic_icenode_model: StochasticMechanisticICENODE,
        segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return stochastic_mechanistic_icenode_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def stochastic_mechanistic_icenode_predictions(stochastic_mechanistic_icenode_model: StochasticMechanisticICENODE,
                                               segmented_admission: SegmentedAdmission,
                                               stochastic_mechanistic_icenode_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return stochastic_mechanistic_icenode_model(admission=segmented_admission,
                                                embedded_admission=stochastic_mechanistic_icenode_embedded_admission,
                                                precomputes=Precomputes())


@pytest.fixture
def koopman_model(in_model_config: ICENODEConfig,
                  in_model_embeddings_config: InICENODEEmbeddingsConfig,
                  dx_scheme: CodingScheme,
                  observation_scheme: NumericScheme,
                  icu_proc_scheme: CodingScheme, hosp_proc_scheme: CodingScheme,
                  icu_inputs_grouping_data) -> InKoopman:
    return InKoopman(config=in_model_config, embeddings_config=in_model_embeddings_config,
                     lead_times=(1.,),
                     dx_codes_size=len(dx_scheme),
                     icu_inputs_grouping=icu_inputs_grouping_data,
                     icu_procedures_size=len(icu_proc_scheme),
                     hosp_procedures_size=len(hosp_proc_scheme),
                     demographic_size=None,
                     observables_size=len(observation_scheme),
                     key=jrandom.PRNGKey(0))


@pytest.fixture
def koopman_embedded_admission(
        koopman_model: InKoopman,
        segmented_admission: SegmentedAdmission) -> EmbeddedAdmission:
    return koopman_model.f_emb(segmented_admission, None)


@pytest.fixture
@pytest.mark.usefixtures('jax_cpu_execution')
def inkoopman_predictions(koopman_model: InKoopman,
                                               segmented_admission: SegmentedAdmission,
                                               koopman_embedded_admission: EmbeddedAdmission) -> AdmissionPrediction:
    return koopman_model(admission=segmented_admission,
                         embedded_admission=koopman_embedded_admission,
                         precomputes=koopman_model.precomputes())


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_inicenode_predictions(inicenode_predictions):
    assert isinstance(inicenode_predictions, AdmissionTrajectoryPrediction)
    assert inicenode_predictions.leading_observable.value.shape == inicenode_predictions.admission.leading_observable.value.shape


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_gru_ode_bayes_predictions(gru_ode_bayes_predictions):
    assert isinstance(gru_ode_bayes_predictions, AdmissionGRUODEBayesPrediction)
    assert gru_ode_bayes_predictions.leading_observable.value.shape == gru_ode_bayes_predictions.admission.leading_observable.value.shape
    assert gru_ode_bayes_predictions.imputed_observables.value.shape == gru_ode_bayes_predictions.admission.observables.value.shape


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_icenode_icnn_predictions(icenode_icnn_predictions: AdmissionPrediction):
    assert isinstance(icenode_icnn_predictions, AdmissionPrediction)
    assert icenode_icnn_predictions.leading_observable.value.shape == icenode_icnn_predictions.admission.leading_observable.value.shape

@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_auto_icnn_predictions(auto_icnn_predictions: AdmissionPrediction):
    assert isinstance(auto_icnn_predictions, AdmissionPrediction)
    assert auto_icnn_predictions.observables.value.shape == auto_icnn_predictions.admission.observables.value.shape



@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_stochastic_icenode_predictions(stochastic_icenode_predictions: AdmissionPrediction):
    assert isinstance(stochastic_icenode_predictions, AdmissionPrediction)
    assert stochastic_icenode_predictions.leading_observable.value.shape == stochastic_icenode_predictions.admission.leading_observable.value.shape


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_stochastic_mechanistic_icenode_predictions(stochastic_mechanistic_icenode_predictions: AdmissionPrediction):
    assert isinstance(stochastic_mechanistic_icenode_predictions, AdmissionPrediction)
    assert stochastic_mechanistic_icenode_predictions.leading_observable.value.shape == stochastic_mechanistic_icenode_predictions.admission.leading_observable.value.shape

@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_inkoopman_predictions(inkoopman_predictions: AdmissionPrediction):
    assert isinstance(inkoopman_predictions, AdmissionPrediction)
    assert inkoopman_predictions.leading_observable.value.shape == inkoopman_predictions.admission.leading_observable.value.shape


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_inicenode_model_grad_apply(inicenode_model: InICENODE, segmented_admission: SegmentedAdmission,
                                    inicenode_embedded_admission: EmbeddedAdmission):
    if len(segmented_admission.leading_observable) == 0:
        pytest.skip("No leading observable")

    def forward(model: InICENODE) -> jnp.ndarray:
        p = model(admission=segmented_admission, embedded_admission=inicenode_embedded_admission,
                  precomputes=Precomputes())
        return jnp.mean(p.leading_observable.value.mean())

    grad = eqx.filter_grad(forward)(inicenode_model)
    assert not tree_hasnan(grad)


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_gru_ode_bayes_model_grad_apply(gru_ode_bayes_model: GRUODEBayes, segmented_admission: SegmentedAdmission,
                                        gru_ode_bayes_embedded_admission: EmbeddedAdmission):
    if len(segmented_admission.leading_observable) == 0:
        pytest.skip("No leading observable")

    def forward(model: GRUODEBayes) -> jnp.ndarray:
        prediction = model(admission=segmented_admission,
                           embedded_admission=gru_ode_bayes_embedded_admission,
                           precomputes=Precomputes())
        p = AdmissionsPrediction().add(subject_id='test_subj', prediction=prediction)
        loss1 = ProbObsPredictionLoss(loss_key='log_normal')(p)
        loss2 = ImputedProbObsPredictionLoss(loss_key='kl_gaussian')(p)
        return loss1 + loss2

    grad = eqx.filter_grad(forward)(gru_ode_bayes_model)
    assert not tree_hasnan(grad)


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_stochastic_icenode_model_grad_apply(stochastic_icenode_model: StochasticInICENODELite,
                                             segmented_admission: SegmentedAdmission,
                                             stochastic_icenode_embedded_admission: EmbeddedAdmission):
    if len(segmented_admission.leading_observable) == 0:
        pytest.skip("No leading observable")

    def forward(model: StochasticInICENODELite) -> jnp.ndarray:
        prediction = model(admission=segmented_admission,
                           embedded_admission=stochastic_icenode_embedded_admission,
                           precomputes=Precomputes())
        p = AdmissionsPrediction().add(subject_id='test_subj', prediction=prediction)
        return ObsPredictionLoss(loss_key='mse')(p)

    grad = eqx.filter_grad(forward)(stochastic_icenode_model)
    assert not tree_hasnan(grad)


@pytest.mark.serial_test
@pytest.mark.usefixtures('jax_cpu_execution')
def test_koopman_model_grad_apply(koopman_model: InKoopman,
                                  segmented_admission: SegmentedAdmission,
                                  koopman_embedded_admission: EmbeddedAdmission):
    if len(segmented_admission.leading_observable) == 0:
        pytest.skip("No leading observable")

    def forward(model: InKoopman) -> jnp.ndarray:
        prediction = model(admission=segmented_admission,
                           embedded_admission=koopman_embedded_admission,
                           precomputes=model.precomputes())
        p = AdmissionsPrediction().add(subject_id='test_subj', prediction=prediction)
        return ObsPredictionLoss(loss_key='mse')(p)

    grad = eqx.filter_grad(forward)(koopman_model)
    assert not tree_hasnan(grad)

# @pytest.fixture
# def intervention_uncertainty_weighting() -> InterventionUncertaintyWeightingScheme:
#     return InterventionUncertaintyWeightingScheme(lead_times=(1.,), leading_observable_index=0)


# @pytest.mark.usefixtures('jax_cpu_execution')
# @pytest.fixture
# def intervention_uncertainty_weighting_out(inicenode_model: InICENODE,
#                                            inicenode_predictions: AdmissionTrajectoryPrediction,
#                                            intervention_uncertainty_weighting: InterventionUncertaintyWeightingScheme,
#                                            segmented_admission: SegmentedAdmission,
#                                            embedded_admission: EmbeddedAdmission) -> Tuple[
#     Tuple[float, int], InpatientObservables]:
#     init_state = inicenode_predictions.trajectory.imputed_state
#     return intervention_uncertainty_weighting(f_obs_decoder=inicenode_model.f_obs_dec,
#                                               f_ode_dyn=inicenode_model.f_dyn,
#                                               initial_states=init_state,
#                                               admission=segmented_admission,
#                                               embedded_admission=embedded_admission,
#                                               precomputes=Precomputes())

# @pytest.mark.serial_test
# @pytest.mark.usefixtures('jax_cpu_execution')
# def test_intervention_uncertainty_weighting_out(intervention_uncertainty_weighting_out):
#     assert len(intervention_uncertainty_weighting_out) == 2
