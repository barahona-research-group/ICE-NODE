"""."""

import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor

import random
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
import logging

import pandas as pd
import dask.dataframe as dd
import numpy as np

from ..utils import load_config, translate_path

from . import outcome as O
from . import coding_scheme as C
from .concept import StaticInfo, Subject, Admission
from .inpatient_concepts import (InpatientInput, InpatientObservables,
                                 Inpatient, InpatientAdmission, CodesVector,
                                 InpatientStaticInfo, AggregateRepresentation,
                                 InpatientInterventions)

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')

StrDict = Dict[str, str]


def try_compute(df):
    if hasattr(df, 'compute'):
        return df.compute()
    else:
        return df


@dataclass
class OutlierRemover:
    c_value: str
    c_code_index: str
    min_val: pd.Series
    max_val: pd.Series

    def __call__(self, df):
        min_val = df[self.c_code_index].map(self.min_val)
        max_val = df[self.c_code_index].map(self.max_val)
        df = df[df[self.c_value].between(min_val, max_val)]
        return df


@dataclass
class ZScoreScaler:
    c_value: str
    c_code_index: str
    mean: pd.Series
    std: pd.Series

    def __call__(self, df):
        mean = df[self.c_code_index].map(self.mean)
        std = df[self.c_code_index].map(self.std)
        df.loc[:, self.c_value] = (df[self.c_value] - mean) / std
        return df


@dataclass
class MaxScaler:
    c_value: str
    c_code_index: str
    max_val: pd.Series

    def __call__(self, df):
        max_val = df[self.c_code_index].map(self.max_val)
        df.loc[:, self.c_value] = df[self.c_value] / max_val
        return df


class AbstractEHRDataset(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_meta_json(cls, meta_fpath, **init_kwargs):
        pass

    @abstractmethod
    def to_subjects(self):
        pass


class MIMIC4EHRDataset(AbstractEHRDataset):

    def __init__(self, df: Dict[str, pd.DataFrame], code_scheme: Dict[str,
                                                                      StrDict],
                 normalised_scheme: StrDict, code_colname: StrDict,
                 code_version_colname: StrDict, adm_colname: StrDict,
                 static_colname: StrDict, name: str, **kwargs):

        normalised_scheme = {
            code_type: eval(f"C.{scheme}")()
            for code_type, scheme in normalised_scheme.items()
        }
        code_scheme = {
            code_type: {
                version: eval(f"C.{scheme}")()
                for version, scheme in version_map.items()
            }
            for code_type, version_map in code_scheme.items()
        }
        for code_type, version_map in code_scheme.items():
            if df.get(code_type, None) is None:
                continue

            c_code = code_colname[code_type]
            c_version = code_version_colname[code_type]

            for version, scheme in version_map.items():
                if isinstance(scheme, C.ICDCommons):
                    ver_mask = df[code_type][c_version].astype(str) == version
                    df[code_type].loc[ver_mask, c_code] = df[code_type].loc[
                        ver_mask, c_code].apply(scheme.add_dots)

            df[code_type] = self._validate_codes(df[code_type], c_code,
                                                 c_version, version_map)

        self.name = name
        self.normalised_scheme = normalised_scheme
        self.code_scheme = code_scheme
        self.adm_colname = adm_colname
        self.static_colname = static_colname
        self.code_colname = code_colname
        self.code_version_colname = code_version_colname
        self.df = df

    @staticmethod
    def load_dataframes(meta):
        m = meta
        files = m['files']
        base_dir = m['base_dir']
        adm_colname = m['adm_colname']
        code_colname = m['code_colname']
        code_ver_colname = m.get('code_version_colname')

        adm_df = pd.read_csv(os.path.join(base_dir, files['adm']))
        static_df = pd.read_csv(os.path.join(base_dir, files['static']))

        # Cast columns of dates to datetime64
        adm_df[adm_colname['admittime']] = pd.to_datetime(
            adm_df[adm_colname['admittime']],
            infer_datetime_format=True).dt.normalize()
        adm_df[adm_colname['dischtime']] = pd.to_datetime(
            adm_df[adm_colname['dischtime']],
            infer_datetime_format=True).dt.normalize()

        res = {'adm': adm_df, 'static': static_df}

        for code_type in ['dx', 'pr']:
            if files.get(code_type, None) is None:
                res[code_type] = None
                continue
            dtype = {code_colname[code_type]: str}
            if code_ver_colname:
                dtype[code_ver_colname[code_type]] = str
            res[code_type] = pd.read_csv(os.path.join(base_dir,
                                                      files[code_type]),
                                         dtype=dtype)

        return res

    @staticmethod
    def _validate_codes(df, code_col, version_col, version_map):
        drop_idx = []
        for version, version_df in df.groupby(version_col):
            codeset = set(version_df[code_col])
            scheme = version_map[str(version)]
            scheme_codes = set(scheme.codes)

            unrecognised = codeset - scheme_codes
            if len(unrecognised) > 0:
                logging.warning(f"""
                    Unrecognised {type(scheme)} codes ({len(unrecognised)})
                    to be removed: {sorted(unrecognised)}""")

            # Data Loss!
            drop_idx.extend(
                version_df[~version_df[code_col].isin(scheme_codes)].index)

        return df.drop(index=drop_idx)

    def to_subjects(self):
        col = self.adm_colname
        scol = self.static_colname
        adm_id_col = col["admission_id"]
        admt_col = col["admittime"]
        dist_col = col["dischtime"]
        dx_scheme = self.normalised_scheme['dx']
        pr_scheme = self.normalised_scheme.get('pr', C.NullScheme())

        s_df = self.df['static']
        gender_dict = {'M': 1.0, 'F': 0.0}

        gender_col = s_df[scol["gender"]].map(gender_dict)
        subject_gender = dict(zip(s_df[scol["subject_id"]], gender_col))
        anchor_date = pd.to_datetime(s_df[scol['anchor_year']],
                                     format='%Y').dt.normalize()
        anchor_age = s_df[scol['anchor_age']].apply(
            lambda y: pd.DateOffset(years=-y))
        dob = anchor_date + anchor_age
        subject_dob = dict(zip(s_df[scol["subject_id"]], dob))

        subjects = {}
        # Admissions
        for subj_id, subj_adms_df in self.df['adm'].groupby(col["subject_id"]):
            subj_adms = {}

            for idx, adm_row in subj_adms_df.iterrows():
                adm_id = adm_row[adm_id_col]
                subj_adms[adm_id] = dict(admission_id=adm_id,
                                         admission_dates=(adm_row[admt_col],
                                                          adm_row[dist_col]),
                                         dx_codes=set(),
                                         pr_codes=set(),
                                         dx_scheme=dx_scheme,
                                         pr_scheme=pr_scheme)
            static_info = StaticInfo(date_of_birth=subject_dob[subj_id],
                                     gender=subject_gender[subj_id])
            subjects[subj_id] = dict(subject_id=subj_id,
                                     admissions=subj_adms,
                                     static_info=static_info)

        for subj_id, adm_id, dx_codes in self.codes_extractor("dx"):
            subjects[subj_id]["admissions"][adm_id]["dx_codes"] = dx_codes

        for subj_id, adm_id, pr_codes in self.codes_extractor("pr"):
            subjects[subj_id]["admissions"][adm_id]["pr_codes"] = pr_codes

        for subj in subjects.values():
            subj['admissions'] = [
                Admission(**adm) for adm in subj['admissions'].values()
            ]
        return [Subject(**subj) for subj in subjects.values()]

    def codes_extractor(self, code_type):
        if any(code_type not in d
               for d in (self.code_colname, self.code_scheme, self.df)):
            return
        if self.normalised_scheme.get(code_type,
                                      C.NullScheme()) is C.NullScheme():
            return

        adm_id_col = self.adm_colname["admission_id"]
        subject_id_col = self.adm_colname["subject_id"]

        code_col = self.code_colname[code_type]
        version_col = self.code_version_colname[code_type]
        version_map = self.code_scheme[code_type]
        t_sch = self.normalised_scheme[code_type]
        df = self.df[code_type]
        for subj_id, subj_df in df.groupby(subject_id_col):
            for adm_id, codes_df in subj_df.groupby(adm_id_col):
                codeset = set()
                for version, version_df in codes_df.groupby(version_col):
                    s_sch = version_map[str(version)]
                    m = s_sch.mapper_to(t_sch)
                    codeset.update(m.map_codeset(version_df[code_col]))

                yield subj_id, adm_id, codeset

    @classmethod
    def from_meta_json(cls, meta_fpath, **init_kwargs):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        meta['df'] = cls.load_dataframes(meta)
        return cls(**meta, **init_kwargs)


class MIMIC3EHRDataset(MIMIC4EHRDataset):

    def __init__(self, df, code_scheme, code_colname, adm_colname,
                 static_colname, name, **kwargs):
        code_scheme = {
            code_type: eval(f"C.{scheme}")()
            for code_type, scheme in code_scheme.items()
        }

        for code_type, scheme in code_scheme.items():
            if df.get(code_type, None) is None:
                continue
            c_code = code_colname[code_type]
            df[code_type].loc[:, c_code] = df[code_type].loc[:, c_code].apply(
                scheme.add_dots)
            df[code_type] = self._validate_codes(df[code_type], c_code, scheme)

        self.name = name
        self.df = df
        self.code_scheme = code_scheme
        self.adm_colname = adm_colname
        self.static_colname = static_colname
        self.code_colname = code_colname

    @staticmethod
    def _validate_codes(df, code_col, scheme):
        drop_idx = []
        codeset = set(df[code_col])
        scheme_codes = set(scheme.codes)
        unrecognised = codeset - scheme_codes
        if len(unrecognised) > 0:
            logging.warning(f"""
                Unrecognised {type(scheme)} codes ({len(unrecognised)})
                to be removed: {sorted(unrecognised)}""")

        # Data Loss!
        drop_idx = df[~df[code_col].isin(scheme_codes)].index

        return df.drop(index=drop_idx)

    def codes_extractor(self, code_type):
        if any(code_type not in d
               for d in (self.code_colname, self.code_scheme, self.df)):
            return

        adm_id_col = self.adm_colname["admission_id"]
        subject_id_col = self.adm_colname["subject_id"]

        code_col = self.code_colname[code_type]
        df = self.df[code_type]
        for subj_id, subj_df in df.groupby(subject_id_col):
            for adm_id, codes_df in subj_df.groupby(adm_id_col):
                yield subj_id, adm_id, set(codes_df[code_col])

    def to_subjects(self):
        col = self.adm_colname
        scol = self.static_colname

        adm_id_col = col["admission_id"]
        admt_col = col["admittime"]
        dist_col = col["dischtime"]
        dx_scheme = self.code_scheme['dx']
        pr_scheme = self.code_scheme.get('pr', C.NullScheme())

        s_df = self.df['static']
        gender_dict = {'M': 1.0, 'F': 0.0}

        gender_col = s_df[scol["gender"]].map(gender_dict)
        subject_gender = dict(zip(s_df[scol["subject_id"]], gender_col))
        dob = pd.to_datetime(s_df[scol["date_of_birth"]],
                             infer_datetime_format=True).dt.normalize()
        subject_dob = dict(zip(s_df[scol["subject_id"]], dob))
        subjects = {}
        # Admissions
        for subj_id, subj_adms_df in self.df['adm'].groupby(col["subject_id"]):
            subj_adms = {}

            for idx, adm_row in subj_adms_df.iterrows():
                adm_id = adm_row[adm_id_col]
                subj_adms[adm_id] = dict(admission_id=adm_id,
                                         admission_dates=(adm_row[admt_col],
                                                          adm_row[dist_col]),
                                         dx_codes=set(),
                                         pr_codes=set(),
                                         dx_scheme=dx_scheme,
                                         pr_scheme=pr_scheme)
            static_info = StaticInfo(gender=subject_gender[subj_id],
                                     date_of_birth=subject_dob[subj_id])
            subjects[subj_id] = dict(subject_id=subj_id,
                                     admissions=subj_adms,
                                     static_info=static_info)

        for subj_id, adm_id, dx_codes in self.codes_extractor("dx"):
            subjects[subj_id]["admissions"][adm_id]["dx_codes"] = dx_codes

        for subj_id, adm_id, pr_codes in self.codes_extractor("pr"):
            subjects[subj_id]["admissions"][adm_id]["pr_codes"] = pr_codes

        for subj in subjects.values():
            subj['admissions'] = [
                Admission(**adm) for adm in subj['admissions'].values()
            ]
        return [Subject(**subj) for subj in subjects.values()]


class CPRDEHRDataset(AbstractEHRDataset):

    def __init__(self, df, colname, code_scheme, name, **kwargs):

        self.name = name
        self.code_scheme = {
            code_type: eval(f"C.{scheme}")()
            for code_type, scheme in code_scheme.items()
        }
        self.colname = colname
        self.df = df

    def to_subjects(self):
        listify = lambda s: list(map(lambda e: e.strip(), s.split(',')))
        col = self.colname
        subjects = {}

        gender_dict = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1])
        }
        gender_missing = np.array([0, 0, 0])

        # Admissions
        for subj_id, subj_df in self.df.groupby(col["subject_id"]):

            assert len(subj_df) == 1, "Each patient should have a single row"

            codes = listify(subj_df.iloc[0][col["dx"]])
            year_month = listify(subj_df.iloc[0][col["year_month"]])

            # To infer date-of-birth
            age0 = int(float(listify(subj_df.iloc[0][col["age"]])[0]))
            year_month0 = pd.to_datetime(
                year_month[0], infer_datetime_format=True).normalize()
            year_of_birth = year_month0 + pd.DateOffset(years=-age0)
            gender_key = int(subj_df.iloc[0][col["gender"]])
            gender = gender_dict.get(gender_key, gender_missing)

            imd = int(subj_df.iloc[0][col["imd_decile"]])
            ethnicity = subj_df.iloc[0][col["ethnicity"]]
            static_info = StaticInfo(ethnicity=ethnicity,
                                     ethnicity_scheme=self.code_scheme["eth"],
                                     date_of_birth=year_of_birth,
                                     gender=gender,
                                     idx_deprivation=imd)

            # codes aggregated by year-month.
            dx_codes_ym_agg = defaultdict(set)
            for code, ym in zip(codes, year_month):
                ym = pd.to_datetime(ym, infer_datetime_format=True).normalize()
                dx_codes_ym_agg[ym].add(code)

            admissions = []
            for adm_idx, adm_date in enumerate(sorted(dx_codes_ym_agg.keys())):
                disch_date = adm_date + pd.DateOffset(days=1)
                dx_codes = dx_codes_ym_agg[adm_date]
                admissions.append(
                    Admission(admission_id=adm_idx,
                              admission_dates=(adm_date, disch_date),
                              dx_codes=dx_codes,
                              pr_codes=set(),
                              dx_scheme=self.code_scheme["dx"],
                              pr_scheme=C.NullScheme()))
            subjects[subj_id] = Subject(subject_id=subj_id,
                                        admissions=admissions,
                                        static_info=static_info)

        return list(subjects.values())

    @classmethod
    def from_meta_json(cls, meta_fpath, **init_kwargs):
        meta = load_config(meta_fpath)
        filepath = translate_path(meta['filepath'])
        meta['df'] = pd.read_csv(filepath, sep='\t', dtype=str)
        return cls(**meta, **init_kwargs)


@dataclass
class MIMIC4ICUDatasetScheme:
    dx_source: Dict[str, C.ICDCommons]
    dx_target: C.ICDCommons
    outcome: O.OutcomeExtractor
    int_proc_source: C.MIMIC4Procedures
    int_proc_target: C.MIMIC4ProcedureGroups
    int_input_source: C.MIMIC4Input
    int_input_target: C.MIMIC4InputGroups
    eth_source: C.MIMIC4Eth
    eth_target: C.MIMIC4Eth
    obs: C.MIMIC4Observables

    def __init__(self, code_scheme: Dict[str, Any]):
        self.dx_source = {
            version: eval(f"C.{scheme}")()
            for version, scheme in code_scheme["dx"][0].items()
        }
        self.dx_target = eval(f"C.{code_scheme['dx'][1]}")()
        outcome_scheme_str = code_scheme["outcome"]
        self.outcome = eval(f"O.OutcomeExtractor(\'{outcome_scheme_str}\')")
        self.int_proc_source = eval(f"C.{code_scheme['int_proc'][0]}")()
        self.int_proc_target = eval(f"C.{code_scheme['int_proc'][1]}")()
        self.int_input_source = eval(f"C.{code_scheme['int_input'][0]}")()
        self.int_input_target = eval(f"C.{code_scheme['int_input'][1]}")()
        self.eth_source = eval(f"C.{code_scheme['ethnicity'][0]}")()
        self.eth_target = eval(f"C.{code_scheme['ethnicity'][1]}")()
        self.obs = eval(f"C.{code_scheme['obs'][0]}")()

    def dx_mapper(self, version):
        return self.dx_source[version].mapper_to(self.dx_target)

    def eth_mapper(self):
        return self.eth_source.mapper_to(self.eth_target)


@dataclass
class MIMIC4ICUDataset(AbstractEHRDataset):
    colname: Dict[str, Dict[str, str]]
    name: str
    df: Dict[str, dd.DataFrame]
    scheme: MIMIC4ICUDatasetScheme
    preprocessing_history: List[Dict[str, Any]]

    def __init__(self, df, colname, code_scheme, name, **kwargs):
        self.name = name
        self.scheme = MIMIC4ICUDatasetScheme(code_scheme)
        # deepcopy to avoid modifying the original colname which is used by
        # reference in the lazy evaluation of dask operations.
        self.colname = copy.deepcopy(colname)
        self.df = df

        self.preprocessing_history = []

        logging.debug("Dataframes validation and time conversion")
        self._dx_fix_icd_dots()
        self._dx_filter_unsupported_icd()

        def _filter_codes(df, c_code, source_scheme):
            mask = df[c_code].isin(source_scheme.codes)
            logging.debug(
                f'Removed codes: {df[~mask][c_code].unique().compute()}')
            return df[mask]

        self.df["int_proc"] = _filter_codes(self.df["int_proc"],
                                            self.colname["int_proc"]["code"],
                                            self.scheme.int_proc_source)
        self.df["int_input"] = _filter_codes(self.df["int_input"],
                                             self.colname["int_input"]["code"],
                                             self.scheme.int_input_source)
        self.df["obs"] = _filter_codes(self.df["obs"],
                                       self.colname["obs"]["code"],
                                       self.scheme.obs)

        self.df["int_proc"] = self._add_code_source_index(
            self.df["int_proc"], self.scheme.int_proc_source,
            self.colname["int_proc"])
        self.df["int_input"] = self._add_code_source_index(
            self.df["int_input"], self.scheme.int_input_source,
            self.colname["int_input"])
        self.df["obs"] = self._add_code_source_index(self.df["obs"],
                                                     self.scheme.obs,
                                                     self.colname["obs"])

        seconds_scaler = 1 / 3600.0  # convert seconds to hours
        self._adm_add_adm_interval(seconds_scaler)
        self._adm_remove_subjects_with_negative_adm_interval()
        self._set_relative_times(self.df, self.colname, "int_proc",
                                 ["start_time", "end_time"], seconds_scaler)
        self._set_relative_times(self.df, self.colname, "int_input",
                                 ["start_time", "end_time"], seconds_scaler)
        self._set_relative_times(self.df,
                                 self.colname,
                                 "obs", ["timestamp"],
                                 seconds_scaler=seconds_scaler)
        self.df = {k: try_compute(v) for k, v in self.df.items()}
        logging.debug("[DONE] Dataframes validation and time conversion")

    def _dx_fix_icd_dots(self):
        c_code = self.colname["dx"]["code"]
        c_version = self.colname["dx"]["version"]
        df = self.df["dx"]
        df = df.assign(**{c_code: df[c_code].str.strip()})
        self.df['dx'] = df
        for version, scheme in self.scheme.dx_source.items():
            if isinstance(scheme, C.ICDCommons):
                ver_mask = df[c_version] == version
                code_col = df[c_code]
                df = df.assign(**{
                    c_code:
                    code_col.mask(ver_mask, code_col.map(scheme.add_dots))
                })
        self.df["dx"] = df

    def _dx_filter_unsupported_icd(self):
        c_code = self.colname["dx"]["code"]
        c_version = self.colname["dx"]["version"]
        self.df["dx"] = self._validate_dx_codes(self.df["dx"], c_code,
                                                c_version,
                                                self.scheme.dx_source)

    @staticmethod
    def _validate_dx_codes(df, code_col, version_col, version_map):
        filtered_df = []
        groups = df.groupby(version_col)
        for version in groups[version_col].unique():
            version_df = groups.get_group(version[0])
            codeset = set(version_df[code_col])
            scheme = version_map[str(version[0])]
            scheme_codes = set(scheme.codes)

            unrecognised = codeset - scheme_codes
            if len(unrecognised) > 0:
                logging.info(
                    f'Unrecognised ICD v{version} codes: {len(unrecognised)} ({len(unrecognised)/len(codeset):.2%})'
                )
                logging.debug(f"""
                    Unrecognised {type(scheme)} codes ({len(unrecognised)})
                    to be removed (first 30): {sorted(unrecognised)[:30]}""")

            filtered_df.append(
                version_df[version_df[code_col].isin(scheme_codes)])

        return dd.concat(filtered_df).reset_index(drop=True)

    @staticmethod
    def _add_code_source_index(df, source_scheme, colname):
        c_code = colname["code"]
        colname["code_source_index"] = "code_source_index"
        df = df.assign(
            code_source_index=df[c_code].map(source_scheme.index).astype(int))
        return df

    def _adm_add_adm_interval(self, seconds_scaler=1 / 3600.0):
        c_admittime = self.colname["adm"]["admittime"]
        c_dischtime = self.colname["adm"]["dischtime"]

        df = self.df["adm"]
        delta = df[c_dischtime] - df[c_admittime]
        df = df.assign(adm_interval=delta.dt.total_seconds() * seconds_scaler)
        self.colname["adm"]["adm_interval"] = "adm_interval"
        self.df["adm"] = df

    def _adm_remove_subjects_with_negative_adm_interval(self):
        c_adm_interval = self.colname["adm"]["adm_interval"]
        c_subject = self.colname["adm"]["subject_id"]

        df = self.df["adm"]
        subjects_neg_intervals = df[df[c_adm_interval] < 0][c_subject].unique()
        logging.debug(
            f"Removing subjects with at least one negative adm_interval: {len(subjects_neg_intervals)}"
        )
        df = df[~df[c_subject].isin(subjects_neg_intervals)]
        self.df["adm"] = df

    @staticmethod
    def _set_relative_times(df_dict,
                            colname,
                            df_name,
                            time_cols,
                            seconds_scaler=1 / 3600.0):
        c_admittime = colname["adm"]["admittime"]
        c_adm_id = colname[df_name]["admission_id"]

        target_df = df_dict[df_name]
        adm_df = df_dict["adm"][[c_admittime]]
        df = target_df.merge(adm_df,
                             left_on=c_adm_id,
                             right_index=True,
                             how='left')

        for time_col in time_cols:
            col = colname[df_name][time_col]
            delta = df[col] - df[c_admittime]
            target_df = target_df.assign(
                **{col: delta.dt.total_seconds() * seconds_scaler})

        df_dict[df_name] = target_df

    def _fit_obs_preprocessing(self, admission_ids, outlier_q1, outlier_q2,
                               outlier_iqr_scale, outlier_z1, outlier_z2):
        c_code_index = self.colname["obs"]["code_source_index"]
        c_value = self.colname["obs"]["value"]
        c_adm_id = self.colname["obs"]["admission_id"]
        df = self.df['obs'][[c_code_index, c_value, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]
        outlier_q = np.array([outlier_q1, outlier_q2])
        q = df.groupby(c_code_index).apply(
            lambda x: x[c_value].quantile(outlier_q))
        q.columns = ['q1', 'q2']
        q['iqr'] = q['q2'] - q['q1']
        q['out_q1'] = q['q1'] - outlier_iqr_scale * q['iqr']
        q['out_q2'] = q['q2'] + outlier_iqr_scale * q['iqr']

        z = df.groupby(c_code_index).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))
        z['out_z1'] = z['mu'] - outlier_z1 * z['sigma']
        z['out_z2'] = z['mu'] + outlier_z2 * z['sigma']

        zscaler = ZScoreScaler(c_value=c_value,
                               c_code_index=c_code_index,
                               mean=z['mu'],
                               std=z['sigma'])
        remover = OutlierRemover(c_value=c_value,
                                 c_code_index=c_code_index,
                                 min_val=np.minimum(q['out_q1'], z['out_z1']),
                                 max_val=np.maximum(q['out_q2'], z['out_z2']))
        return {'scaler': zscaler, 'outlier_remover': remover}

    def _fit_int_input_processing(self, admission_ids):
        c_adm_id = self.colname["int_input"]["admission_id"]
        c_code_index = self.colname["int_input"]["code_source_index"]
        c_rate = self.colname["int_input"]["rate"]
        df = self.df["int_input"][[c_adm_id, c_code_index, c_rate]]
        df = df[df[c_adm_id].isin(admission_ids)]
        return {
            'scaler':
            MaxScaler(c_value=c_rate,
                      c_code_index=c_code_index,
                      max_val=df.groupby(c_code_index).apply(
                          lambda x: x[c_rate].max()))
        }

    def fit_preprocessing(self,
                          subject_ids: List[int] = None,
                          outlier_q1=0.25,
                          outlier_q2=0.75,
                          outlier_iqr_scale=1.5,
                          outlier_z1=-2.5,
                          outlier_z2=2.5):

        c_subject = self.colname["adm"]["subject_id"]
        adm_df = self.df["adm"]
        if subject_ids is None:
            train_adms = adm_df.index
        else:
            train_adms = adm_df[adm_df[c_subject].isin(subject_ids)].index
        preprocessor = {}
        preprocessor['obs'] = self._fit_obs_preprocessing(
            admission_ids=train_adms,
            outlier_q1=outlier_q1,
            outlier_q2=outlier_q2,
            outlier_iqr_scale=outlier_iqr_scale,
            outlier_z1=outlier_z1,
            outlier_z2=outlier_z2)
        preprocessor['int_input'] = self._fit_int_input_processing(train_adms)
        return preprocessor

    def apply_preprocessing(self, preprocessor):
        assert len(self.preprocessing_history) == 0, \
            "Preprocessing can only be applied once."

        self.preprocessing_history.append(preprocessor)
        for df_name, _preprocessor in preprocessor.items():
            if 'outlier_remover' in _preprocessor:
                remover = _preprocessor['outlier_remover']
                n1 = len(self.df[df_name])
                self.df[df_name] = remover(self.df[df_name])
                n2 = len(self.df[df_name])
                logging.debug(
                    f'Removed {n1 - n2} ({(n1 - n2) / n2 :0.3f}) outliers from {df_name}'
                )
            if 'scaler' in _preprocessor:
                scaler = _preprocessor['scaler']
                self.df[df_name] = scaler(self.df[df_name])

    @staticmethod
    def load_dataframes(meta):
        files = meta['files']
        base_dir = meta['base_dir']
        colname = meta['colname']

        dtype = {
            **{
                colname["dx"]["code"]: str,
                colname["dx"]["version"]: str
            },
            **{
                colname[f]["admission_id"]: int
                for f in files.keys() if "admission_id" in colname[f]
            },
            **{
                colname[f]["subject_id"]: int
                for f in files.keys() if "subject_id" in colname[f]
            }
        }

        logging.debug('Loading dataframe files')
        df = {
            k:
            dd.read_csv(os.path.join(base_dir, files[k]),
                        usecols=colname[k].values(),
                        dtype=dtype)
            for k in files.keys()
        }
        logging.debug('[DONE] Loading dataframe files')

        # admission_id matching
        logging.debug("Matching admission_id")
        df["adm"] = df["adm"].set_index(colname["adm"]["index"]).compute()

        df_with_adm_id = {
            name: df[name]
            for name in df if "admission_id" in colname[name]
        }
        df_with_adm_id = {
            name: _df[_df[colname[name]["admission_id"]].isin(df["adm"].index)]
            for name, _df in df_with_adm_id.items()
        }
        df_with_adm_id = {
            name: _df.reset_index()
            for name, _df in df_with_adm_id.items()
        }
        df.update(df_with_adm_id)
        logging.debug("[DONE] Matching admission_id")

        logging.debug("Time casting..")
        # Cast timestamps for admissions
        for time_col in ("admittime", "dischtime"):
            df["adm"][colname["adm"][time_col]] = dd.to_datetime(
                df["adm"][colname["adm"][time_col]],
                infer_datetime_format=True)

        # Cast timestamps for intervensions
        for time_col in ("start_time", "end_time"):
            for file in ("int_proc", "int_input"):
                df[file][colname[file][time_col]] = dd.to_datetime(
                    df[file][colname[file][time_col]],
                    infer_datetime_format=True)

        # Cast timestamps for observables
        df["obs"][colname["obs"]["timestamp"]] = dd.to_datetime(
            df["obs"][colname["obs"]["timestamp"]], infer_datetime_format=True)
        logging.debug("[DONE] Time casting..")
        return df

    @classmethod
    def from_meta_json(cls, meta_fpath, **init_kwargs):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        meta['df'] = cls.load_dataframes(meta)
        return cls(**meta, **init_kwargs)

    def to_subjects(self, subject_ids, num_workers):

        subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
            subject_ids)
        admission_ids = self.adm_extractor(subject_ids)
        adm_ids_list = sum(map(list, admission_ids.values()), [])
        logging.debug('Extracting dx codes...')
        dx_codes = dict(self.dx_codes_extractor(adm_ids_list))
        logging.debug('[DONE] Extracting dx codes')
        logging.debug('Extracting dx codes history...')
        dx_codes_history = dict(
            self.dx_codes_history_extractor(dx_codes, admission_ids))
        logging.debug('[DONE] Extracting dx codes history')
        logging.debug('Extracting outcome...')
        outcome = dict(self.outcome_extractor(dx_codes))
        logging.debug('[DONE] Extracting outcome')
        logging.debug('Extracting procedures...')
        procedures = dict(self.procedure_extractor(adm_ids_list))
        logging.debug('[DONE] Extracting procedures')
        logging.debug('Extracting inputs...')
        inputs = dict(self.inputs_extractor(adm_ids_list))
        logging.debug('[DONE] Extracting inputs')
        logging.debug('Extracting observables...')
        observables = dict(
            self.observables_extractor(adm_ids_list, num_workers))
        logging.debug('[DONE] Extracting observables')

        logging.debug('Compiling admissions...')
        c_admittime = self.colname['adm']['admittime']
        c_dischtime = self.colname['adm']['dischtime']
        c_adm_interval = self.colname['adm']['adm_interval']
        adf = self.df['adm']
        adm_dates = dict(
            zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))
        proc_repr = AggregateRepresentation(self.scheme.int_proc_source,
                                            self.scheme.int_proc_target)

        def gen_admission(i):
            adm_interval = adf.loc[i, c_adm_interval].item()
            interventions = InpatientInterventions(proc=procedures[i],
                                                   input_=inputs[i],
                                                   adm_interval=adm_interval)
            interventions = interventions.segment_proc(proc_repr)
            obs = observables[i].segment(interventions.time)
            return InpatientAdmission(admission_id=i,
                                      admission_dates=adm_dates[i],
                                      dx_codes=dx_codes[i],
                                      dx_codes_history=dx_codes_history[i],
                                      outcome=outcome[i],
                                      observables=obs,
                                      interventions=interventions)

        def _gen_subject(subject_id):

            _admission_ids = admission_ids[subject_id]
            # for subject_id, subject_admission_ids in admission_ids.items():
            _admission_ids = sorted(_admission_ids)

            static_info = InpatientStaticInfo(
                date_of_birth=subject_dob[subject_id],
                gender=subject_gender[subject_id],
                ethnicity=subject_eth[subject_id],
                ethnicity_scheme=self.scheme.eth_target)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                admissions = list(executor.map(gen_admission, _admission_ids))
            return Inpatient(subject_id=subject_id,
                             admissions=admissions,
                             static_info=static_info)

        return list(map(_gen_subject, subject_ids))

    @property
    def subject_ids(self):
        return sorted(
            self.df['adm'][self.colname['adm']['subject_id']].unique())

    def random_splits(self,
                      splits: List[float],
                      random_seed: int = 42,
                      balanced: str = 'subjects'):
        rng = random.Random(random_seed)
        subject_ids = self.subject_ids
        rng.shuffle(subject_ids)
        subject_ids = np.array(subject_ids)

        c_subject_id = self.colname['adm']['subject_id']
        c_adm_interval = self.colname['adm']['adm_interval']
        adm_df = self.df['adm']

        if balanced == 'subjects':
            probs = (np.ones(len(subject_ids)) / len(subject_ids)).cumsum()

        elif balanced == 'admissions':
            n_adms = adm_df.groupby(c_subject_id).size()
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

    def subject_info_extractor(self, subject_ids):

        static_df = self.df['static']
        c_s_subject_id = self.colname["static"]["subject_id"]
        c_gender = self.colname["static"]["gender"]
        c_anchor_year = self.colname["static"]["anchor_year"]
        c_anchor_age = self.colname["static"]["anchor_age"]
        c_eth = self.colname["static"]["ethnicity"]

        static_df = static_df[static_df[c_s_subject_id].isin(subject_ids)]
        gender_column = static_df[c_gender]
        subject_gender = dict(zip(static_df[c_s_subject_id], gender_column))

        anchor_date = pd.to_datetime(static_df[c_anchor_year],
                                     format='%Y').dt.normalize()
        anchor_age = static_df[c_anchor_age].map(
            lambda y: pd.DateOffset(years=-y))
        dob = anchor_date + anchor_age
        subject_dob = dict(zip(static_df[c_s_subject_id], dob))
        subject_eth = dict()
        eth_mapper = self.scheme.eth_mapper()
        for subject_id, subject_df in static_df.groupby(c_s_subject_id):
            eth_code = eth_mapper.map_codeset(subject_df[c_eth].tolist())
            subject_eth[subject_id] = eth_mapper.codeset2vec(eth_code)

        return subject_dob, subject_gender, subject_eth

    def adm_extractor(self, subject_ids):
        c_subject_id = self.colname["adm"]["subject_id"]
        df = self.df["adm"]
        df = df[df[c_subject_id].isin(subject_ids)]
        return {
            subject_id: subject_df.index.tolist()
            for subject_id, subject_df in df.groupby(c_subject_id)
        }

    def dx_codes_extractor(self, admission_ids_list):
        c_adm_id = self.colname["dx"]["admission_id"]
        c_code = self.colname["dx"]["code"]
        c_version = self.colname["dx"]["version"]

        df = self.df["dx"]
        df = df[df[c_adm_id].isin(admission_ids_list)]
        codes_df = {
            adm_id: codes_df
            for adm_id, codes_df in df.groupby(c_adm_id)
        }
        empty_codes = CodesVector.empty(self.scheme.dx_target)

        def _extract_codes(adm_id):
            _codes_df = codes_df.get(adm_id)
            if _codes_df is None:
                return (adm_id, empty_codes)

            vec = np.zeros(len(self.scheme.dx_target), dtype=bool)
            for version, version_df in _codes_df.groupby(c_version):
                mapper = self.scheme.dx_mapper(str(version))
                codeset = mapper.map_codeset(version_df[c_code])
                vec = np.maximum(vec, mapper.codeset2vec(codeset))
            return (adm_id, CodesVector(vec, self.scheme.dx_target))

        return map(_extract_codes, admission_ids_list)

    def dx_codes_history_extractor(self, dx_codes, admission_ids):
        for subject_id, subject_admission_ids in admission_ids.items():
            _adm_ids = sorted(subject_admission_ids)
            vec = np.zeros(len(self.scheme.dx_target), dtype=bool)
            yield _adm_ids[0], CodesVector(vec, self.scheme.dx_target)

            for prev_adm_id, adm_id in zip(_adm_ids[:-1], _adm_ids[1:]):
                if prev_adm_id in dx_codes:
                    vec = np.maximum(vec, dx_codes[prev_adm_id].vec)
                yield adm_id, CodesVector(vec, self.scheme.dx_target)

    def outcome_extractor(self, dx_codes):

        def _extract_outcome(adm_id):
            _dx_codes = dx_codes[adm_id]
            outcome_codes = self.scheme.outcome.map_codeset(
                _dx_codes.to_codeset(), self.scheme.dx_target)
            outcome_vec = self.scheme.outcome.codeset2vec(
                outcome_codes, self.scheme.dx_target)
            return (adm_id, CodesVector(outcome_vec, self.scheme.outcome))

        return map(_extract_outcome, dx_codes.keys())

    def procedure_extractor(self, admission_ids_list):
        c_adm_id = self.colname["int_proc"]["admission_id"]
        c_code_index = self.colname["int_proc"]["code_source_index"]
        c_start_time = self.colname["int_proc"]["start_time"]
        c_end_time = self.colname["int_proc"]["end_time"]
        df = self.df["int_proc"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        def group_fun(x):
            return pd.Series({
                0: x[c_code_index].to_numpy(),
                1: x[c_start_time].to_numpy(),
                2: x[c_end_time].to_numpy()
            })

        grouped = df.groupby(c_adm_id).apply(group_fun)
        adm_arr = grouped.index.tolist()
        input_size = len(self.scheme.int_proc_source)
        for i in adm_arr:
            yield (i,
                   InpatientInput(index=grouped.loc[i, 0],
                                  rate=np.ones_like(grouped.loc[i, 0],
                                                    dtype=bool),
                                  starttime=grouped.loc[i, 1],
                                  endtime=grouped.loc[i, 2],
                                  size=input_size))

        for adm_id in set(admission_ids_list) - set(adm_arr):
            yield (adm_id, InpatientInput.empty(input_size))

    def inputs_extractor(self, admission_ids_list):
        c_adm_id = self.colname["int_input"]["admission_id"]
        c_start_time = self.colname["int_input"]["start_time"]
        c_end_time = self.colname["int_input"]["end_time"]
        c_rate = self.colname["int_input"]["rate"]
        c_code_index = self.colname["int_input"]["code_source_index"]

        df = self.df["int_input"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        def group_fun(x):
            return pd.Series({
                0: x[c_code_index].to_numpy(),
                1: x[c_rate].to_numpy().astype(np.float16),
                2: x[c_start_time].to_numpy(),
                3: x[c_end_time].to_numpy()
            })

        grouped = df.groupby(c_adm_id).apply(group_fun)
        adm_arr = grouped.index.tolist()
        input_size = len(self.scheme.int_input_source)
        for i in adm_arr:
            yield (i,
                   InpatientInput(index=grouped.loc[i, 0],
                                  rate=grouped.loc[i, 1],
                                  starttime=grouped.loc[i, 2],
                                  endtime=grouped.loc[i, 3],
                                  size=input_size))
        for adm_id in set(admission_ids_list) - set(adm_arr):
            yield (adm_id, InpatientInput.empty(input_size))

    def observables_extractor(self, admission_ids_list, num_workers):
        c_adm_id = self.colname["obs"]["admission_id"]
        c_time = self.colname["obs"]["timestamp"]
        c_value = self.colname["obs"]["value"]
        c_code_index = self.colname["obs"]["code_source_index"]

        df = self.df["obs"][[c_adm_id, c_time, c_value, c_code_index]]
        logging.debug("obs: filter adms")
        df = df[df[c_adm_id].isin(admission_ids_list)]

        obs_dim = len(self.scheme.obs)

        def ret_put(a, *args):
            np.put(a, *args)
            return a

        def val_mask(x):
            idx = x[c_code_index]
            val = ret_put(np.empty(obs_dim, dtype=np.float16), idx, x[c_value])
            mask = ret_put(np.zeros(obs_dim, dtype=bool), idx, 1.0)
            adm_id = x.index[0]
            time = x[c_time].iloc[0].item()
            return pd.Series({0: adm_id, 1: time, 2: val, 3: mask})

        def gen_observation(val_mask):
            time = val_mask[1].to_numpy()
            value = val_mask[2]
            mask = val_mask[3]
            mask = np.vstack(mask.values).reshape((len(time), obs_dim))
            value = np.vstack(value.values).reshape((len(time), obs_dim))
            return InpatientObservables(time=time, value=value, mask=mask)

        def partition_fun(part_df):
            g = part_df.groupby([c_adm_id, c_time], sort=True, as_index=False)
            return g.apply(val_mask).groupby(0).apply(gen_observation)

        logging.debug("obs: dasking")
        df = df.set_index(c_adm_id)
        df = dd.from_pandas(df, npartitions=12, sort=True)
        logging.debug("obs: groupby")
        obs_obj_df = df.map_partitions(partition_fun, meta=(None, object))
        logging.debug("obs: undasking")
        obs_obj_df = obs_obj_df.compute()
        logging.debug("obs: extract")

        collected_adm_ids = obs_obj_df.index.tolist()
        assert len(collected_adm_ids) == len(set(collected_adm_ids)), \
            "Duplicate admission ids in obs"

        for adm_id, obs in obs_obj_df.items():
            yield (adm_id, obs)

        logging.debug("obs: empty")
        for adm_id in set(admission_ids_list) - set(obs_obj_df.index):
            yield (adm_id, InpatientObservables.empty(obs_dim))


def load_dataset(label, **init_kwargs):

    if label == 'M3':
        return MIMIC3EHRDataset.from_meta_json(f'{_META_DIR}/mimic3_meta.json',
                                               **init_kwargs)
    if label == 'M4':
        return MIMIC4EHRDataset.from_meta_json(f'{_META_DIR}/mimic4_meta.json',
                                               **init_kwargs)
    if label == 'CPRD':
        return CPRDEHRDataset.from_meta_json(f'{_META_DIR}/cprd_meta.json',
                                             **init_kwargs)
    if label == 'M4ICU':
        return MIMIC4ICUDataset.from_meta_json(
            f'{_META_DIR}/mimic4icu_meta.json', **init_kwargs)
