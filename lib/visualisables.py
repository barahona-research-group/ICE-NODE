from __future__ import annotations
from typing import List, Optional, Union, Dict, Any, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
import equinox as eqx

from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import Span
from bokeh.palettes import tol

from .base import Data, Module
from .ehr import (Admission, InpatientObservables,
                  InpatientInterventions, DemographicVectorConfig,
                  LeadingObservableConfig, TVxEHR, PatientTrajectory,
                  DatasetScheme, AbstractScheme)
from .ml.artefacts import AdmissionPrediction, Predictions, TrajectoryConfig
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
        for label, traj in self.obs_pred_trajectory.items():
            dfs[label] = pd.DataFrame({'time': traj.time, 'value': traj.value})
        return dfs

    @property
    def lead_dataframe(self):
        dfs = []
        for label, lead in self.lead.items():
            dfs.append(
                pd.DataFrame({
                    'time': lead.time,
                    'value': lead.value,
                    'label': label
                }))
        if len(dfs) == 0:
            return None
        return pd.concat(dfs)

    @property
    def lead_pred_trajectory_dataframe(self):
        dfs = []
        for label, traj in sorted(self.lead_pred_trajectory.items()):
            dfs.append(
                pd.DataFrame({
                    'time': traj.time,
                    'value': traj.value,
                    'label': label
                }))
        return pd.concat(dfs)

    @property
    def interventions_dataframe(self):
        df = []
        for order, interval in enumerate(self.interventions):
            description = []
            for (code, value) in interval.input_list:
                description.append(f"{code} ({value:.2f})")
            for code in interval.proc_list:
                description.append(code)
            description = ', '.join(description)
            df.append({
                'order': order,
                'description': description,
                't0': interval.t0,
                't1': interval.t1
            })
        return pd.DataFrame(df)


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

    @property
    def lead_index2label(self):
        index = list(range(len(self.lead_times)))
        return dict(zip(index, self.lead_times))

    @property
    def obs_index2label(self):
        return self.obs_scheme.index2desc

    @property
    def obs_minmax(self):
        min_dict = self.scalers_history['obs'].min_val.to_dict()
        max_dict = self.scalers_history['obs'].max_val.to_dict()
        return {
            self.obs_index2label[i]: (min_dict[i], max_dict[i])
            for i in range(len(self.obs_index2label))
        }

    @property
    def lead_minmax(self):
        obs_index = self.leading_observable_config.index
        obs_label = self.obs_index2label[obs_index]
        return self.obs_minmax[obs_label]

    @property
    def lead_times(self):
        return self.leading_observable_config.leading_hours

    @property
    def lead_colors(self):
        return tol['Light'][len(self.lead_times)]

    @property
    def lead_label_colors(self):
        return dict(zip(self.lead_times, self.lead_colors))

    @property
    def int_input_index2label(self):
        return self.int_input_scheme.index2desc

    @property
    def int_proc_index2label(self):
        return self.int_proc_scheme.index2desc

    def __call__(self, model: InpatientModel, prediction: AdmissionPrediction):

        def _aux(obs, lead, interventions, obs_traj, lead_traj, unscale=True):
            if unscale:
                obs = self.obs_unscaler(obs)
                lead = self.lead_unscaler(lead)
                interventions = self.input_unscaler(interventions)
                obs_traj = self.obs_unscaler(obs_traj)
                lead_traj = self.lead_unscaler(lead_traj)

            obs = obs.groupby_code(self.obs_index2label)
            lead = lead.groupby_code(self.lead_index2label)
            obs_traj = obs_traj.groupby_code(self.obs_index2label)
            lead_traj = lead_traj.groupby_code(self.lead_index2label)
            input_list = IntervalInterventions.from_interventions(
                interventions,
                inputs_index2code=self.int_input_index2label,
                proc_index2code=self.int_proc_index2label)
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

    def batch_predict(self, model: InpatientModel, inpatients: TVxEHR):
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
        obs_timestamps = visualisables.observation_timestamps
        time = visualisables.trajectory_timesamples
        traj = visualisables.obs_pred_trajectory_dataframes
        obs_minmax = self.obs_minmax
        for code, obs in visualisables.obs_dataframes.items():
            p = figure(width=600,
                       height=300,
                       title=code,
                       x_axis_label='Time since admission (hrs)')
            p.xaxis.ticker = time
            p.y_range.start = obs_minmax[code][0]
            p.y_range.end = obs_minmax[code][1]
            p.x_range.start = 0
            p.x_range.end = min(time[-1], 72)

            p.scatter(x='time',
                      y='value',
                      source=obs,
                      size=9,
                      alpha=0.6,
                      legend_label='Observed')
            p.line(x='time',
                   y='value',
                   source=traj[code],
                   color="navy",
                   alpha=0.4,
                   line_width=4,
                   legend_label='Predicted')
            p.vspan(x=obs_timestamps, line_color='#009E73', line_width=1)
            figures[code] = p

        return figures

    def _sort_lead(self, visualisables: AdmissionVisualisables):
        lead_df = visualisables.lead_dataframe

        lead_df = lead_df.groupby(['time', 'value'],
                                  as_index=False).agg({'label': 'min'})
        lead_df['color'] = lead_df['label'].map(self.lead_label_colors)
        lead_df['label'] = lead_df['label'].astype(int)
        lead_df['caption'] = lead_df['label'].map(
            lambda x: f'AKI max-stage in {x} hrs')
        return lead_df.sort_values(by='label')

    def _make_bokeh_lead(self, visualisables: AdmissionVisualisables):
        if len(visualisables.lead) == 0:
            return None

        groundtruth_timestamps = visualisables.observation_timestamps
        groundtruth_df = self._sort_lead(visualisables)
        obs_key = self.obs_index2label[self.leading_observable_config.index]
        current_obs = visualisables.obs_dataframes[obs_key]
        # remove ground-truth lead where an actual observation exists
        # and the value is the same
        mask = groundtruth_df.apply(lambda x: any(
            (x.time == current_obs['time']) &
            (x.value == current_obs['value'])),
                                    axis=1)
        groundtruth_df = groundtruth_df[~mask]

        prediction_time = visualisables.trajectory_timesamples
        prediction_traj = visualisables.lead_pred_trajectory_dataframe
        prediction_traj['color'] = prediction_traj['label'].map(
            self.lead_label_colors)
        prediction_traj['label'] = prediction_traj['label'].astype(int)

        p = figure(width=900,
                   height=300,
                   title='AKI Stage (Ground Truth and Prediction)',
                   x_axis_label='Time since admission (hrs)')

        p.xaxis.ticker = prediction_time
        p.y_range.start = self.lead_minmax[0]
        p.y_range.end = self.lead_minmax[1]
        p.x_range.start = 0
        p.x_range.end = min(prediction_time[-1], 36)

        p.scatter(x='time',
                  y='value',
                  line_color='black',
                  color='color',
                  source=groundtruth_df,
                  size=16,
                  alpha=0.6,
                  legend_field='caption')

        p.scatter(x='time',
                  y='value',
                  line_color='red',
                  color='red',
                  marker='plus',
                  source=current_obs,
                  size=18,
                  alpha=1.0,
                  legend_label='Current observation')
        p.text(x='time',
               y='value',
               text='label',
               source=groundtruth_df,
               text_font_size='10pt',
               text_align='center',
               text_baseline='middle')
        for (label, color), df in prediction_traj.groupby(['label', 'color']):
            p.line(x='time',
                   y='value',
                   line_color=color,
                   source=df,
                   alpha=1.0,
                   line_width=2,
                   legend_label=f'Predicted AKI max-stage in {label} hrs')
        p.vspan(x=groundtruth_timestamps, line_color='#009E73', line_width=0.5)
        return p

    def _make_bokeh_interventions(self, visualisables: AdmissionVisualisables):
        df = visualisables.interventions_dataframe
        df['adjusted_order'] = min(10, len(df)) - df['order']
        p = figure(width=900,
                   height=300,
                   title='Interventions',
                   x_axis_label='Time since admission (hrs)')
        time = visualisables.trajectory_timesamples
        p.xaxis.ticker = time
        p.yaxis.visible = False
        p.y_range.start = 0
        p.y_range.end = min(10, len(df))
        p.x_range.start = 0
        p.x_range.end = min(time[-1], 36)
        p.hbar(y="adjusted_order",
               left='t0',
               right='t1',
               height=0.6,
               source=df)
        p.text(x='t1',
               y='adjusted_order',
               text='description',
               source=df,
               text_font_size='12pt',
               text_align='left',
               text_baseline='middle')
        return p

    def make_bokeh(self, visualisables: AdmissionVisualisables):
        return {
            'obs': column(list(self._make_bokeh_obs(visualisables).values())),
            'lead': self._make_bokeh_lead(visualisables),
            'interventions': self._make_bokeh_interventions(visualisables)
        }
