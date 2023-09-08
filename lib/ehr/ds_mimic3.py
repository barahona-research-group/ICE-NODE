"""."""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Dict, List, Optional, ClassVar
from concurrent.futures import ThreadPoolExecutor
import random
import logging

import pandas as pd
import dask.dataframe as dd
import numpy as np

from .coding_scheme import (AbstractScheme, Ethnicity)
from .concepts import (Patient, Admission, DemographicVectorConfig, StaticInfo)
from .dataset import Dataset, DatasetScheme


def try_compute(df):
    if hasattr(df, 'compute'):
        return df.compute()
    else:
        return df


class MIMIC3Dataset(Dataset):
    df: Dict[str, dd.DataFrame]
    scheme: DatasetScheme
    seconds_scaler: ClassVar[float] = 1 / 3600.0  # convert seconds to hours

    @classmethod
    def sample_n_subjects(cls, df, c_subject_id, n, seed=None):
        if seed is not None:
            rng = random.Random(seed)

        subjects = df[c_subject_id].unique().compute()
        subjects = rng.sample(subjects.tolist(), n)
        return df[df[c_subject_id].isin(subjects)]

    @classmethod
    def _match_admissions_with_demographics(cls, df, colname):
        adm = df["adm"]
        static = df["static"]
        c_subject_id = colname["adm"].subject_id
        subject_ids = list(set(adm[c_subject_id].unique()) & set(static.index))
        logging.debug(
            f"Removing subjects by matching demographic"
            f"(-{len(set(static.index) - set(subject_ids))})"
            f"and admissions"
            f"(-{len(set(adm[c_subject_id].unique()) - set(subject_ids))})"
            "tables")

        static = static.loc[subject_ids]
        adm = adm[adm[c_subject_id].isin(subject_ids)]
        df["adm"] = adm
        df["static"] = static

    def _load_dataframes(self):
        config = self.config.copy()
        files = config.files
        colname = self.colname
        logging.debug('Loading dataframe files')
        df = {
            k:
            dd.read_csv(os.path.join(config.path, files[k]),
                        usecols=colname[k].columns,
                        dtype=colname[k].default_raw_types)
            for k in files.keys()
        }
        if config.sample is not None:
            df["adm"] = self.sample_n_subjects(df["adm"],
                                               colname["adm"].subject_id,
                                               config.sample, 0)
        logging.debug('[DONE] Loading dataframe files')
        logging.debug('Preprocess admissions')
        df["adm"] = df["adm"].compute()
        df["static"] = df["static"].compute()

        df["static"] = df["static"].set_index(colname["static"].index)
        df["adm"] = df["adm"].set_index(colname["adm"].index)
        self._match_admissions_with_demographics(df, colname)
        adm = df["adm"]

        adm = self._adm_cast_times(adm, colname["adm"])
        adm, colname["adm"] = self._adm_add_adm_interval(
            adm, colname["adm"], self.seconds_scaler)
        adm = self._adm_remove_subjects_with_negative_adm_interval(
            adm, colname["adm"])
        adm, merger_map = self._adm_handle_overlapping_admissions(
            adm, colname["adm"], action=config.overlapping_admissions)

        logging.debug('[DONE] Preprocess admissions')

        # admission_id matching
        logging.debug("Matching admission_id")
        df_with_adm_id = {
            name: df[name]
            for name in df if colname[name].has('admission_id')
        }
        df_with_adm_id = self._map_admission_ids(df_with_adm_id, colname,
                                                 merger_map)
        df_with_adm_id = self._match_filter_admission_ids(
            adm, df_with_adm_id, colname)
        df.update(df_with_adm_id)
        logging.debug("[DONE] Matching admission_id")

        df["adm"] = adm

        logging.debug("Dataframes validation and time conversion")
        self.df = {k: try_compute(v) for k, v in df.items()}
        self._dx_fix_icd_dots()
        self._dx_filter_unsupported_icd()
        self.colname = colname
        self._match_admissions_with_demographics(self.df, colname)
        logging.debug("[DONE] Dataframes validation and time conversion")

    def to_subjects(self, subject_ids: List[int], num_workers: int,
                    demographic_vector_config: DemographicVectorConfig,
                    target_scheme: DatasetScheme, **kwargs):

        subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
            subject_ids, target_scheme)
        admission_ids = self.adm_extractor(subject_ids)
        adm_ids_list = sum(map(list, admission_ids.values()), [])
        logging.debug('Extracting dx codes...')
        dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
        logging.debug('[DONE] Extracting dx codes')
        logging.debug('Extracting dx codes history...')
        dx_codes_history = dict(
            self.dx_codes_history_extractor(dx_codes, admission_ids,
                                            target_scheme))
        logging.debug('[DONE] Extracting dx codes history')
        logging.debug('Extracting outcome...')
        outcome = dict(self.outcome_extractor(dx_codes, target_scheme))
        logging.debug('[DONE] Extracting outcome')

        logging.debug('Compiling admissions...')
        c_admittime = self.colname['adm'].admittime
        c_dischtime = self.colname['adm'].dischtime
        adf = self.df['adm']
        adm_dates = dict(
            zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))

        def gen_admission(i):
            return Admission(admission_id=i,
                             admission_dates=adm_dates[i],
                             dx_codes=dx_codes[i],
                             dx_codes_history=dx_codes_history[i],
                             outcome=outcome[i],
                             observables=None,
                             interventions=None)

        def _gen_subject(subject_id):

            _admission_ids = admission_ids[subject_id]
            # for subject_id, subject_admission_ids in admission_ids.items():
            _admission_ids = sorted(_admission_ids,
                                    key=lambda aid: adm_dates[aid][0])
            static_info = StaticInfo(
                date_of_birth=subject_dob[subject_id],
                gender=subject_gender[subject_id],
                ethnicity=subject_eth[subject_id],
                demographic_vector_config=demographic_vector_config)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                admissions = list(executor.map(gen_admission, _admission_ids))
            return Patient(subject_id=subject_id,
                           admissions=admissions,
                           static_info=static_info)

        return list(map(_gen_subject, subject_ids))

    def _dx_fix_icd_dots(self):
        c_code = self.colname["dx"].code
        add_dots = self.scheme.dx.add_dots
        df = self.df["dx"]
        df = df.assign(**{c_code: df[c_code].str.strip().map(add_dots)})
        self.df['dx'] = df

    def _dx_filter_unsupported_icd(self):
        c_code = self.colname["dx"].code
        df = self.df["dx"]
        codeset = set(df[c_code])
        scheme = self.scheme.dx
        scheme_codes = set(scheme.codes)

        unrecognised = codeset - scheme_codes
        if len(unrecognised) > 0:
            logging.debug(f'Unrecognised ICD codes: {len(unrecognised)} '
                          f'({len(unrecognised)/len(codeset):.2%})')
            logging.debug(f'Unrecognised {type(scheme)} codes '
                          f'({len(unrecognised)}) '
                          f'to be removed (first 30): '
                          f'{sorted(unrecognised)[:30]}')
        df = df[df[c_code].isin(scheme_codes)]
        self.df['dx'] = df

    def random_splits(self,
                      splits: List[float],
                      subject_ids: Optional[List[int]] = None,
                      random_seed: int = 42,
                      balanced: str = 'subjects',
                      ignore_first_admission: bool = False):
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_ids = sorted(subject_ids)

        random.Random(random_seed).shuffle(subject_ids)
        subject_ids = np.array(subject_ids)

        c_subject_id = self.colname['adm'].subject_id
        c_adm_interval = self.colname['adm'].adm_interval
        adm_df = self.df['adm']
        adm_df = adm_df[adm_df[c_subject_id].isin(subject_ids)]

        if balanced == 'subjects':
            probs = (np.ones(len(subject_ids)) / len(subject_ids)).cumsum()

        elif balanced == 'admissions':
            n_adms = adm_df.groupby(c_subject_id).size()
            if ignore_first_admission:
                n_adms = n_adms - 1
            p_adms = n_adms.loc[subject_ids] / n_adms.sum()
            probs = p_adms.values.cumsum()

        elif balanced == 'adm_interval':
            subject_intervals_sum = adm_df.groupby(c_subject_id).agg(
                total_interval=(c_adm_interval, 'sum'))
            p_subject_intervals = subject_intervals_sum.loc[
                subject_ids] / subject_intervals_sum.sum()
            probs = p_subject_intervals.values.cumsum()

        splits = np.searchsorted(probs, splits)
        return [a.tolist() for a in np.split(subject_ids, splits)]

    @staticmethod
    def _adm_handle_overlapping_admissions(adm_df,
                                           colname,
                                           action='merge',
                                           interval_inclusive=True):
        logging.debug("adm: Merging overlapping admissions")
        c_subject_id = colname.subject_id
        c_admittime = colname.admittime
        c_dischtime = colname.dischtime
        df = adm_df.copy()

        # Step 1: Collect overlapping admissions
        def _collect_overlaps(s_df):
            s_df = s_df.sort_values(c_admittime)
            disch_cummax = s_df.iloc[:-1][c_dischtime].cummax()
            disch_cummax_idx = disch_cummax.map(lambda x:
                                                (x == disch_cummax).idxmax())

            if interval_inclusive:
                has_parent = s_df[c_admittime].iloc[
                    1:].values <= disch_cummax.values
            else:
                has_parent = s_df[c_admittime].iloc[
                    1:].values < disch_cummax.values

            s_df = s_df.iloc[1:]
            s_df['sup_adm'] = np.where(has_parent, disch_cummax_idx, pd.NA)
            s_df = s_df[s_df.sup_adm.notnull()]
            return s_df['sup_adm'].to_dict()

        subj_df = {sid: sdf for sid, sdf in df.groupby(c_subject_id)}
        ch2pt = {}
        for sid, s_df in subj_df.items():
            ch2pt.update(_collect_overlaps(s_df))
        ch_adms = list(ch2pt.keys())

        if action == 'remove':
            subject_ids = df[df.index.isin(ch_adms)][c_subject_id].unique()
            df = df[~df[c_subject_id].isin(subject_ids)]
            return df, {}
        elif action == 'merge':
            # Step 3: Find super admission, in case of multiple levels of overlap.
            def super_adm(ch):
                while ch in ch2pt:
                    ch = ch2pt[ch]
                return ch

            ch2sup = dict(zip(ch_adms, map(super_adm, ch_adms)))

            # Step 4: reversed map from super admission to child admissions.
            sup2ch = defaultdict(list)
            for ch, sup in ch2sup.items():
                sup2ch[sup].append(ch)

            # List of super admissiosn.
            sup_adms = list(sup2ch.keys())
            # List of admissions to be removed after merging.
            rem_adms = sum(map(list, sup2ch.values()), [])

            # Step 5: Merge overlapping admissions by extending discharge time.
            df['adm_id'] = df.index
            df.loc[sup_adms, c_dischtime] = df.loc[sup_adms].apply(
                lambda x: max(x[c_dischtime], df.loc[sup2ch[x.adm_id],
                                                     c_dischtime].max()),
                axis=1)

            # Step 6: Remove merged admissions.
            df = df.drop(index=rem_adms)
            df = df.drop(columns=['adm_id'])

            logging.debug(
                f"adm: Merged {len(rem_adms)} overlapping admissions")
            return df, ch2sup
        else:
            raise ValueError(f"Unknown action {action}")

    @staticmethod
    def _match_filter_admission_ids(adm_df, dfs, colname):
        dfs = {
            name: _df[_df[colname[name].admission_id].isin(adm_df.index)]
            for name, _df in dfs.items()
        }
        return {name: _df.reset_index() for name, _df in dfs.items()}

    @staticmethod
    def _adm_cast_times(adm_df, colname):
        df = adm_df.copy()
        # Cast timestamps for admissions
        for time_col in (colname.admittime, colname.dischtime):
            df[time_col] = pd.to_datetime(df[time_col])
        return df

    @staticmethod
    def _map_admission_ids(df, colname, merger_map):
        updated = {}
        for name, _df in df.items():
            c_adm_id = colname[name].admission_id
            _df[c_adm_id] = _df[c_adm_id].map(lambda x: merger_map.get(x, x))
            updated[name] = _df
        return updated

    @staticmethod
    def _adm_add_adm_interval(adm_df, colname, seconds_scaler):
        c_admittime = colname.admittime
        c_dischtime = colname.dischtime

        delta = adm_df[c_dischtime] - adm_df[c_admittime]
        adm_df = adm_df.assign(
            adm_interval=(delta.dt.total_seconds() *
                          seconds_scaler).astype(np.float32))
        colname = colname._replace(adm_interval="adm_interval")
        return adm_df, colname

    @staticmethod
    def _adm_remove_subjects_with_negative_adm_interval(adm_df, colname):
        c_adm_interval = colname.adm_interval
        c_subject = colname.subject_id

        subjects_neg_intervals = adm_df[adm_df[c_adm_interval] <
                                        0][c_subject].unique()
        logging.debug(
            f"Removing subjects with at least one negative adm_interval: "
            f"{len(subjects_neg_intervals)}")
        df = adm_df[~adm_df[c_subject].isin(subjects_neg_intervals)]
        return df

    @property
    def subject_ids(self):
        return sorted(self.df["static"].index.unique())

    def subject_info_extractor(self, subject_ids: List[int],
                               target_scheme: DatasetScheme):
        """
        Important comment from MIMIC-III documentation at \
            https://mimic.mit.edu/docs/iii/tables/patients/
        > DOB is the date of birth of the given patient. Patients who are \
            older than 89 years old at any time in the database have had their\
            date of birth shifted to obscure their age and comply with HIPAA.\
            The shift process was as follows: the patient’s age at their \
            first admission was determined. The date of birth was then set to\
            exactly 300 years before their first admission.
        """
        assert self.scheme.gender is target_scheme.gender, (
            "No conversion assumed for gender attribute")

        c_gender = self.colname["static"].gender
        c_eth = self.colname["static"].ethnicity
        c_dob = self.colname["static"].date_of_birth

        c_admittime = self.colname["adm"].admittime
        c_dischtime = self.colname["adm"].dischtime
        c_subject_id = self.colname["adm"].subject_id

        adm_df = self.df['adm'][self.df['adm'][c_subject_id].isin(subject_ids)]

        df = self.df['static'].copy()
        df = df.loc[subject_ids]
        gender = df[c_gender].map(self.scheme.gender.codeset2vec)

        subject_gender = gender.to_dict()

        df[c_dob] = pd.to_datetime(df[c_dob])
        last_disch_date = adm_df.groupby(c_subject_id)[c_dischtime].max()
        first_adm_date = adm_df.groupby(c_subject_id)[c_admittime].min()

        last_disch_date = last_disch_date.loc[df.index]
        first_adm_date = first_adm_date.loc[df.index]
        uncertainty = (last_disch_date.dt.year - first_adm_date.dt.year) // 2
        shift = (uncertainty + 89).astype('timedelta64[Y]')
        df.loc[:, c_dob] = df[c_dob].mask(
            (last_disch_date.dt.year - df[c_dob].dt.year) > 150,
            first_adm_date - shift)

        subject_dob = df[c_dob].dt.normalize().to_dict()
        # TODO: check https://mimic.mit.edu/docs/iii/about/time/
        eth_mapper = self.scheme.ethnicity_mapper(target_scheme)

        def eth2vec(eth):
            code = eth_mapper.map_codeset(eth)
            return eth_mapper.codeset2vec(code)

        subject_eth = df[c_eth].map(eth2vec).to_dict()

        return subject_dob, subject_gender, subject_eth

    def adm_extractor(self, subject_ids):
        c_subject_id = self.colname["adm"].subject_id
        df = self.df["adm"]
        df = df[df[c_subject_id].isin(subject_ids)]
        return {
            subject_id: subject_df.index.tolist()
            for subject_id, subject_df in df.groupby(c_subject_id)
        }

    def dx_codes_extractor(self, admission_ids_list,
                           target_scheme: DatasetScheme):
        c_adm_id = self.colname["dx"].admission_id
        c_code = self.colname["dx"].code

        df = self.df["dx"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        codes_df = {
            adm_id: codes_df
            for adm_id, codes_df in df.groupby(c_adm_id)
        }
        empty_vector = target_scheme.dx.empty_vector()
        mapper = self.scheme.dx_mapper(target_scheme)

        def _extract_codes(adm_id):
            _codes_df = codes_df.get(adm_id)
            if _codes_df is None:
                return (adm_id, empty_vector)
            codeset = mapper.map_codeset(_codes_df[c_code])
            return (adm_id, mapper.codeset2vec(codeset))

        return dict(map(_extract_codes, admission_ids_list))

    def dx_codes_history_extractor(self, dx_codes, admission_ids,
                                   target_scheme):
        for subject_id, subject_admission_ids in admission_ids.items():
            _adm_ids = sorted(subject_admission_ids)
            vec = target_scheme.dx.empty_vector()
            yield (_adm_ids[0], vec)

            for prev_adm_id, adm_id in zip(_adm_ids[:-1], _adm_ids[1:]):
                if prev_adm_id in dx_codes:
                    vec = vec.union(dx_codes[prev_adm_id])
                yield (adm_id, vec)

    def outcome_extractor(self, dx_codes, target_scheme):
        return zip(dx_codes.keys(),
                   map(target_scheme.outcome.mapcodevector, dx_codes.values()))
