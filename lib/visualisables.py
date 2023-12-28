from __future__ import annotations
from typing import List, Optional, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
import equinox as eqx

from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.transform import jitter

from .base import Data, Module
from .ehr import (Admission, AdmissionPrediction, InpatientObservables,
                  InpatientInterventions, DemographicVectorConfig,
                  LeadingObservableConfig, Patients, PatientTrajectory,
                  TrajectoryConfig, DatasetScheme, AbstractScheme, Predictions)
from .ml import InpatientModel


class IntervalInterventions(Data):
    t0: float
    t1: float
    input_list: List[Tuple[str, float]]
    proc_list: List[str]

    @staticmethod
    def from_interventions(interventions: InpatientInterventions,
                           inputs_index2code: Dict[int, str],
                           proc_index2code: Dict[int, str]):
        inputs = interventions.segmented_input
        procs = interventions.segmented_proc

        assert (inputs is not None and len(inputs_index2code) == inputs.shape[1]), \
            (f"Scheme length {len(inputs_index2code)} does not match input length {inputs.shape[1]}")
        assert procs is not None and len(proc_index2code) == procs.shape[1], \
            f"Scheme length {len(proc_index2code)} does not match proc length {procs.shape[1]}"

        listified = []
        for (t0, t1, inp, proc) in zip(interventions.t0, interventions.t1,
                                       inputs, procs):
            nonzero_input = np.argwhere(inp != 0).flatten()
            nonzero_proc = np.argwhere(proc != 0).flatten()
            if len(nonzero_input) == 0 and len(nonzero_proc) == 0:
                continue
            input_list = []
            for i in nonzero_input:
                input_list.append((inputs_index2code[i], inp[i]))
            proc_list = []
            for i in nonzero_proc:
                proc_list.append(proc_index2code[i])
            interval = IntervalInterventions(t0, t1, input_list, proc_list)
            listified.append(interval)
        return listified


class AdmissionVisualisables(Data):

    # The processed dataset has scaled observables and leading observables.
    obs: Dict[str, InpatientObservables]
    lead: Dict[str, InpatientObservables]
    obs_pred_trajectory: Dict[str, PatientTrajectory]
    lead_pred_trajectory: Dict[str, PatientTrajectory]
    interventions: List[IntervalInterventions]

    @property
    def observation_timestamps(self):
        timestamps = set()
        for obs in self.obs.values():
            timestamps.update(obs.time.tolist())
        return sorted(list(timestamps))

    @property
    def trajectory_timesamples(self):
        traj = next(iter(self.obs_pred_trajectory.values()))
        return traj.time[::4]

    @property
    def obs_dataframes(self):
        dfs = {}
        for name, obs in self.obs.items():
            dfs[name] = pd.DataFrame({'time': obs.time, 'value': obs.value})
        return dfs

    @property
    def obs_pred_trajectory_dataframes(self):
        dfs = {}
        for name, traj in self.obs_pred_trajectory.items():
            dfs[name] = pd.DataFrame({'time': traj.time, 'value': traj.value})
        return dfs


class ModelVisualiser(eqx.Module):
    scalers_history: Dict[str, Any]
    trajectory_config: TrajectoryConfig
    obs_scheme: AbstractScheme
    int_input_scheme: AbstractScheme
    int_proc_scheme: AbstractScheme
    leading_observable_config: LeadingObservableConfig

    @property
    def input_unscaler(self):
        f = self.scalers_history['int_input'].unscale
        return lambda x: eqx.tree_at(lambda o: o.segmented_input, x,
                                     f(x.segmented_input))

    @property
    def obs_unscaler(self):
        f = self.scalers_history['obs'].unscale
        return lambda x: eqx.tree_at(lambda o: o.value, x, f(x.value))

    @property
    def lead_unscaler(self):
        obs_index = self.leading_observable_config.index
        f = lambda x: self.scalers_history['obs'].unscale_code(x, obs_index)
        return lambda x: eqx.tree_at(lambda o: o.value, x, f(x.value))

    def __call__(self, model: InpatientModel, prediction: AdmissionPrediction):

        def _aux(obs, lead, interventions, obs_traj, lead_traj, unscale=True):
            if unscale:
                obs = self.obs_unscaler(obs)
                lead = self.lead_unscaler(lead)
                interventions = self.input_unscaler(interventions)
                obs_traj = self.obs_unscaler(obs_traj)
                lead_traj = self.lead_unscaler(lead_traj)

            obs = obs.groupby_code(self.obs_scheme.index2desc)
            lead = lead.groupby_code(self.leading_observable_config.index2desc)
            obs_traj = obs_traj.groupby_code(self.obs_scheme.index2desc)
            lead_traj = lead_traj.groupby_code(
                self.leading_observable_config.index2desc)
            input_list = IntervalInterventions.from_interventions(
                interventions,
                inputs_index2code=self.int_input_scheme.index2desc,
                proc_index2code=self.int_proc_scheme.index2desc)
            return AdmissionVisualisables(obs=obs,
                                          lead=lead,
                                          interventions=input_list,
                                          obs_pred_trajectory=obs_traj,
                                          lead_pred_trajectory=lead_traj)

        adm = prediction.to_cpu().defragment_observables().admission
        traj = prediction.trajectory

        obs_traj = model.decode_obs_trajectory(traj).to_cpu()
        lead_traj = model.decode_lead_trajectory(traj).to_cpu()

        return _aux(adm.observables, adm.leading_observable, adm.interventions,
                    obs_traj, lead_traj)

    def batch_predict(self, model: InpatientModel, inpatients: Patients):
        predictions = model.batch_predict(
            inpatients,
            leave_pbar=False,
            regularisation=None,
            store_embeddings=self.trajectory_config)
        visualisables = {}
        for sid, spreds in predictions.items():
            visualisables[sid] = {}
            for aid, apreds in spreds.items():
                visualisables[sid][aid] = self(model, apreds)

        return visualisables

    def _make_bokeh_obs(self, visualisables: AdmissionVisualisables):
        figures = {}
        time = visualisables.trajectory_timesamples

        for code, obs in visualisables.obs_dataframes.items():
            p = figure(width=600,
                       height=300,
                       title=code,
                       x_axis_label='Time since admission (hrs)')
            p.scatter(x='time', y='value', source=obs, size=9, alpha=0.6)
            p.xaxis.ticker = time
            figures[code] = p

        return figures

    def _make_bokeh_obs_traj(self, visualisables: AdmissionVisualisables):
        figures = {}
        time = visualisables.trajectory_timesamples

        for code, obs in visualisables.obs_pred_trajectory_dataframes.items():
            p = figure(width=600,
                       height=300,
                       title=code,
                       x_axis_label='Time since admission (hrs)')
            p.line(x='time',
                   y='value',
                   source=obs,
                   color="navy",
                   alpha=0.4,
                   line_width=4)
            p.xaxis.ticker = time
            figures[code] = p

        return figures

    def make_bokeh(self, visualisables: AdmissionVisualisables):
        return {
            'obs':
            column(list(self._make_bokeh_obs(visualisables).values())),
            'obs_traj':
            column(list(self._make_bokeh_obs_traj(visualisables).values()))
        }
