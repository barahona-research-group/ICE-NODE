

#
# class TrajectoryInterventionEffectEstimator(NeuralODESolver):
#     f: ForcedVectorFieldPair
#
#     def get_args(self, x0: PyTree, u: PyTree, precomputes: Optional[Precomputes]) -> PyTree:
#         forced, unforced = u
#         return forced, unforced
#
#     def get_aug_x0(self, x0: PyTree, precomputes: Precomputes) -> PyTree:
#         return x0, x0
#
# class ForcedVectorFieldPair(ForcedVectorField):
#
#     def __call__(self, f: Callable[[jnp.ndarray], jnp.ndarray], t: float, x: PyTree, u: PyTree) -> PyTree:
#         force, zero_force = u
#         x_f, x_a = x
#         return f(jnp.hstack((x_f, force))), f(jnp.hstack((x_a, zero_force)))
#
#
#
# class InterventionUncertaintyWeightingScheme(eqx.Module):  # Uncertainty-Aware InICENODE
#     leading_observable_index: int
#     lead_times: Tuple[float, ...]
#
#     def _estimate_intervention_effect(self,
#                                       f_obs_decoder: eqx.nn.MLP,
#                                       f_ua_dyn: TrajectoryInterventionEffectEstimator,
#                                       timestamp_index, initial_state: jnp.ndarray,
#                                       admission: SegmentedAdmission,
#                                       embdedded_interventions: jnp.ndarray,
#                                       embedded_demographic: Optional[jnp.ndarray],
#                                       precomputes: Precomputes) -> Tuple[Tuple[float, ...], Tuple[int, jnp.ndarray]]:
#
#         """
#         Generate supplementary statistics relative to the timestamp_index of the admission observables, which
#         will be used to estimate the intervention effect as an uncertainty proxy for the outcome prediction.
#         The outcome prediction at time t is assumed to match up with the delayed response at
#         admission.leading_observable[timestamp_index], which is a strong assumption due to:
#             - The interventions applied between t and the delayed response time (t + lead_time), can mitigate
#                 the delayed response per se, so it will be non-ideal to penalize a positive outcome prediction
#                  as a false positive, while in fact it is a correct prediction of the outcome.
#             - In contrast, the interventions themselves might solely induce the patient's condition, so it
#                 will be misleading to penalize the model for negative outcome prediction as false negative,
#                 while in fact it is a correct prediction of a missing outcome.
#         So this function will estimate the effect of the interventions by producing:
#             - A maximum absolute difference between (a) the predicted delayed response under interventions
#                 (calling it forced observable response) with (b) the predicted delayed response with interventions
#                 masked-out (calling it autonomous response). The demographics are not considered in the intervention
#                 effect estimation.
#             - The difference between the predicted forced observable response and the ground-truth response. It
#                 will be used to improve the model prediction of the intervention effect.
#         """
#         # Current state value/time.
#         state = initial_state
#         state_t = admission.observables.time[timestamp_index]
#
#         # Collect the observable of interest.
#         observables = admission.observables
#         obs_mask = observables.mask[:, self.leading_observable_index]
#         obs_values = observables.value[obs_mask][:, self.leading_observable_index]
#         obs_times = observables.time[obs_mask]
#
#         # Demographics and masked-out interventions.
#         demo_e = empty_if_none(embedded_demographic)
#         no_intervention_u = jnp.hstack((demo_e, jnp.zeros_like(embdedded_interventions[0])))
#
#         # the defined delays (t+L) for all L in lead_times.
#         delays = tuple(state_t + lead_time for lead_time in self.lead_times)
#         # grid of delays to probe the intervention max effect.
#         delays_grid = tuple(jnp.linspace(d_t0, d_t1, 10) for d_t0, d_t1 in zip((state_t,) + delays[:-1], delays))
#
#         # timestamps where ground-truth observables are available.
#         obs_times = tuple(t for t in obs_times if state_t < t <= delays[-1])
#         forced_obs_pred_diff = tuple()  # forced observable prediction at obs_times.
#         predicted_intervention_effect = tuple()  # Stores max absolute difference between forced and autonomous preds.
#         current_intervention_effect = 0.0  # Stores the last max absolute difference between forced and autonomous preds.
#         for segment_index in range(observables.n_segments):
#             segment_t1 = admission.interventions.t1[segment_index]
#             if state_t < segment_t1:
#                 continue
#             segment_interventions = embdedded_interventions[segment_index]
#             intervention_u = jnp.hstack((demo_e, segment_interventions))
#             segment_delay_grid = tuple(SubSaveAt(ts=delays_grid) for delay_t, delay_grid in
#                                        zip(delays, delays_grid) if state_t < delay_t <= segment_t1)
#             # Limit to 5, less shape variations, less JITs.
#             obs_segment_ts = tuple(t for t in obs_times if state_t < t <= segment_t1)[:5]
#             segment_obs_times = SubSaveAt(ts=obs_segment_ts) if len(obs_segment_ts) > 0 else None
#
#             saveat = SaveAt(subs=(SubSaveAt(t1=True), segment_obs_times, segment_delay_grid))
#             state, obs_ts_state, delay_ts_state = f_ua_dyn(state, t0=state_t, t1=segment_t1, saveat=saveat,
#                                                            u=(intervention_u, no_intervention_u),
#                                                            precomputes=precomputes)
#             if obs_ts_state:
#                 forced_state, _ = obs_ts_state
#                 forced_obs_pred = eqx.filter_vmap(f_obs_decoder)(forced_state)[:, self.leading_observable_index]
#                 forced_obs_pred_diff += (forced_obs_pred.squeeze() - obs_values[obs_times][:5],)
#
#             for delayed_state_grid in delay_ts_state:
#                 forced_state, auto_state = delayed_state_grid
#                 forced_delayed_pred = eqx.filter_vmap(f_obs_decoder)(forced_state)[:, self.leading_observable_index]
#                 auto_delayed_pred = eqx.filter_vmap(f_obs_decoder)(auto_state)[:, self.leading_observable_index]
#                 grid_max_effect = jnp.max(jnp.abs(forced_delayed_pred - auto_delayed_pred))
#                 current_intervention_effect = jnp.maximum(current_intervention_effect, grid_max_effect)
#                 predicted_intervention_effect += (current_intervention_effect,)
#
#             state_t = segment_t1
#             if state_t >= delays[-1]:
#                 break
#
#         forced_prediction_l2 = jnp.mean(jnp.hstack(forced_obs_pred_diff) ** 2)
#         forced_prediction_n = sum(map(len, forced_obs_pred_diff))
#
#         assert len(predicted_intervention_effect) <= len(self.lead_times)
#         predicted_intervention_effect += (current_intervention_effect,) * (
#                 len(self.lead_times) - len(predicted_intervention_effect))
#         return predicted_intervention_effect, (forced_prediction_n, forced_prediction_l2)
#
#     def __call__(self,
#                  f_obs_decoder: eqx.nn.MLP,
#                  f_ode_dyn: NeuralODESolver,
#                  initial_states: jnp.ndarray,
#                  admission: SegmentedAdmission,
#                  embedded_admission: EmbeddedAdmission, precomputes: Precomputes) -> Tuple[
#         Tuple[float, int], InpatientObservables]:
#         intervention_effect = tuple()
#         forced_prediction_l2 = tuple()
#         f_uncertainty_dyn = TrajectoryInterventionEffectEstimator.from_shared_dyn(f_ode_dyn)
#         assert len(admission.observables.time) == len(initial_states)
#         for i, (_, _, mask) in enumerate(admission.leading_observable):
#             if mask.sum() == 0:
#                 intervention_effect += ((0.0,) * len(self.lead_times),)
#                 forced_prediction_l2 += ((0, 0.0),)
#             else:
#                 estimands = self._estimate_intervention_effect(f_obs_decoder, f_uncertainty_dyn,
#                                                                i, initial_states[i],
#                                                                admission, embedded_admission.interventions,
#                                                                embedded_admission.demographic,
#                                                                precomputes)
#                 intervention_effect += (estimands[0],)
#                 forced_prediction_l2 += (estimands[1],)
#         intervention_effect_array = jnp.array(intervention_effect)
#         intervention_effect_struct = InpatientObservables(time=admission.leading_observable.time,
#                                                           value=intervention_effect_array,
#                                                           mask=admission.leading_observable.mask)
#         forced_prediction_l2, n = zip(*forced_prediction_l2)
#         sum_n = sum(n)
#         forced_prediction_l2_mean = sum(l2 * n / sum_n for l2, n in zip(forced_prediction_l2, n))
#         return (forced_prediction_l2_mean, sum_n), intervention_effect_struct
