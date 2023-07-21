"""."""

import os
from pathlib import Path
from typing import Dict
from collections import defaultdict
from absl import logging
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from tqdm import tqdm
import logging

import pandas as pd
import jax.numpy as jnp
import numpy as np

from ..utils import load_config, translate_path

from . import coding_scheme as C
from . import outcome as O
from .concept import StaticInfo, Subject, Admission
from .icu_concepts import (InpatientInput, InpatientObservables, Inpatient,
                           InpatientAdmission, Codes, StaticInfo as
                           InpatientStaticInfo, AggregateRepresentation,
                           InpatientSegmentedInput)

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')

StrDict = Dict[str, str]


class AbstractEHRDataset(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def from_meta_json(cls, meta_fpath):
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
                        ver_mask, c_code].apply(scheme.add_dot)

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
            infer_datetime_format=True).dt.normalize().dt.to_pydatetime()
        adm_df[adm_colname['dischtime']] = pd.to_datetime(
            adm_df[adm_colname['dischtime']],
            infer_datetime_format=True).dt.normalize().dt.to_pydatetime()

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
        anchor_date = pd.to_datetime(
            s_df[scol['anchor_year']],
            format='%Y').dt.normalize().dt.to_pydatetime()
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
    def from_meta_json(cls, meta_fpath):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        meta['df'] = cls.load_dataframes(meta)
        return cls(**meta)


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
                scheme.add_dot)
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
        dob = pd.to_datetime(
            s_df[scol["date_of_birth"]],
            infer_datetime_format=True).dt.normalize().dt.to_pydatetime()
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
    def from_meta_json(cls, meta_fpath):
        meta = load_config(meta_fpath)
        filepath = translate_path(meta['filepath'])
        meta['df'] = pd.read_csv(filepath, sep='\t', dtype=str)
        return cls(**meta)


@dataclass
class MIMIC4ICUDataset(AbstractEHRDataset):
    colname: Dict[str, Dict[str, str]]
    name: str
    df: Dict[str, pd.DataFrame]
    dx_source_scheme: Dict[str, C.ICDCommons]
    dx_target_scheme: C.ICDCommons

    def __init__(self, df, colname, code_scheme, name, **kwargs):
        self.name = name
        self.dx_source_scheme = {
            version: eval(f"C.{scheme}")()
            for version, scheme in code_scheme["dx"][0].items()
        }

        self.colname = colname
        self.df = df
        self.dx_target_scheme = eval(f"C.{code_scheme['dx'][1]}")()
        outcome_scheme_str = code_scheme["outcome"]
        self.outcome_scheme = eval(
            f"O.OutcomeExtractor(\'{outcome_scheme_str}\')")
        self.int_proc_source_scheme = eval(f"C.{code_scheme['int_proc'][0]}")()
        self.int_proc_target_scheme = eval(f"C.{code_scheme['int_proc'][1]}")()
        self.int_input_source_scheme = eval(
            f"C.{code_scheme['int_input'][0]}")()
        self.int_input_target_scheme = eval(
            f"C.{code_scheme['int_input'][1]}")()
        self.eth_source_scheme = eval(f"C.{code_scheme['ethnicity'][0]}")()
        self.eth_target_scheme = eval(f"C.{code_scheme['ethnicity'][1]}")()
        self.obs_scheme = eval(f"C.{code_scheme['obs'][0]}")()

        c_code = colname["dx"]["code"]
        c_version = colname["dx"]["version"]
        for version, scheme in self.dx_source_scheme.items():
            if isinstance(scheme, C.ICDCommons):
                ver_mask = df["dx"][c_version].astype(str) == version
                df["dx"].loc[ver_mask,
                             c_code] = df["dx"].loc[ver_mask, c_code].apply(
                                 scheme.add_dot)

        df["dx"] = self._validate_dx_codes(df["dx"], c_code, c_version,
                                            self.dx_source_scheme)

    @staticmethod
    def load_dataframes(meta):
        files = meta['files']
        base_dir = meta['base_dir']
        colname = meta['colname']
        code_scheme = meta['code_scheme']

        dtype = {
            **{col: str
               for col in colname["dx"].values()},
            **{
                colname[f]["admission_id"]: str
                for f in files.keys() if "admission_id" in colname[f]
            },
            **{
                colname[f]["subject_id"]: str
                for f in files.keys() if "subject_id" in colname[f]
            },
        }

        df = {
            k: pd.read_csv(os.path.join(base_dir, files[k]), dtype=dtype)
            for k in files.keys()
        }
        # Cast timestamps for admissions
        for time_col in ("admittime", "dischtime"):
            df["adm"][colname["adm"][time_col]] = pd.to_datetime(
                df["adm"][colname["adm"][time_col]],
                infer_datetime_format=True).dt.to_pydatetime()

        # Cast timestamps for intervensions
        for time_col in ("start_time", "end_time"):
            for file in ("int_proc", "int_input"):
                df[file][colname[file][time_col]] = pd.to_datetime(
                    df[file][colname[file][time_col]],
                    infer_datetime_format=True).dt.to_pydatetime()

        # Cast timestamps for observables
        for file in df.keys():
            if not file.startswith("obs"):
                continue
            df[file][colname[file]["timestamp"]] = pd.to_datetime(
                df[file][colname[file]["timestamp"]],
                infer_datetime_format=True).dt.to_pydatetime()

        return df

    @classmethod
    def from_meta_json(cls, meta_fpath):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        logging.debug('Loading dataframes')
        meta['df'] = cls.load_dataframes(meta)
        logging.debug('[DONE] Loading dataframes')
        return cls(**meta)

    @staticmethod
    def _validate_dx_codes(df, code_col, version_col, version_map):
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

        subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
        )
        adm_dates, admission_ids = self.adm_extractor()
        n = len(admission_ids)
        dx_codes = {
            k: v
            for k, v in tqdm(
                self.dx_codes_extractor(), total=n, desc="Discharge dx codes")
        }

        dx_codes_history = {
            k: v
            for k, v in tqdm(self.dx_codes_history_extractor(
                dx_codes, admission_ids),
                             total=n,
                             desc="Discharge dx codes history")
        }

        outcome = {
            k: v
            for k, v in tqdm(
                self.outcome_extractor(dx_codes), total=n, desc="Outcomes")
        }

        procedures = {
            k: v
            for k, v in tqdm(
                self.procedure_extractor(), total=n, desc="Procedures")
        }
        inputs = {
            k: v
            for k, v in tqdm(self.inputs_extractor(), total=n, desc="Inputs")
        }
        observables = {
            k: v
            for k, v in tqdm(
                self.observables_extractor(), total=n, desc="Observables")
        }

        def gen_admission(i):
            return InpatientAdmission(admission_id=i,
                                      admission_dates=adm_dates[i],
                                      dx_codes=dx_codes[i],
                                      dx_codes_history=dx_codes_history[i],
                                      outcome=outcome[i],
                                      procedures=procedures[i],
                                      inputs=inputs[i],
                                      observables=observables[i])

        subjects = []
        for subject_id, subject_admission_ids in admission_ids.items():
            subject_admission_ids = sorted(subject_admission_ids)

            subject_admissions = list(map(gen_admission,
                                          subject_admission_ids))
            static_info = InpatientStaticInfo(
                date_of_birth=subject_dob[subject_id],
                gender=subject_gender[subject_id],
                ethnicity=subject_eth[subject_id],
                ethnicity_scheme=self.eth_target_scheme)
            subjects.append(
                Inpatient(subject_id=subject_id,
                          admissions=subject_admissions,
                          static_info=static_info))

        return subjects

    def subject_info_extractor(self):

        static_df = self.df['static']
        c_s_subject_id = self.colname["static"]["subject_id"]
        c_gender = self.colname["static"]["gender"]
        c_anchor_year = self.colname["static"]["anchor_year"]
        c_anchor_age = self.colname["static"]["anchor_age"]
        c_eth = self.colname["static"]["ethnicity"]

        gender_column = static_df[c_gender]
        subject_gender = dict(zip(static_df[c_s_subject_id], gender_column))

        anchor_date = pd.to_datetime(
            static_df[c_anchor_year],
            format='%Y').dt.normalize().dt.to_pydatetime()
        anchor_age = static_df[c_anchor_age].apply(
            lambda y: pd.DateOffset(years=-y))
        dob = anchor_date + anchor_age
        subject_dob = dict(zip(static_df[c_s_subject_id], dob))
        subject_eth = dict()
        eth_mapper = self.eth_source_scheme.mapper(self.eth_target_scheme)
        for subject_id, subject_df in static_df.groupby(c_s_subject_id):
            eth = eth_mapper.codeset2vec(subject_df[c_eth].tolist())
            subject_eth[subject_id] = eth

        return subject_dob, subject_gender, subject_eth

    def adm_extractor(self):
        c_adm_id = self.colname["adm"]["admission_id"]
        c_admittime = self.colname["adm"]["admittime"]
        c_dischtime = self.colname["adm"]["dischtime"]
        c_subject_id = self.colname["adm"]["subject_id"]

        df = self.df["adm"]
        adm_time = dict(
            zip(df[c_adm_id], zip(df[c_admittime], df[c_dischtime])))
        subject_admissions = {}

        for subject_id, subject_df in df.groupby(c_subject_id):
            subject_admissions[subject_id] = subject_df[c_adm_id].tolist()

        return adm_time, subject_admissions

    def dx_codes_extractor(self):
        c_adm_id = self.colname["dx"]["admission_id"]
        c_code = self.colname["dx"]["code"]
        c_version = self.colname["dx"]["version"]

        df = self.df["dx"]
        for adm_id, codes_df in df.groupby(c_adm_id):
            codeset = set()
            vec = jnp.zeros(len(self.dx_target_scheme))
            for version, version_df in codes_df.groupby(c_version):
                src_scheme = self.dx_source_scheme[str(version)]
                mapper = src_scheme.mapper_to(self.dx_target_scheme)
                codeset.update(mapper.map_codeset(version_df[c_code]))
                vec = jnp.maximum(vec, mapper.codeset2vec(codeset))
            yield adm_id, Codes(codeset, vec, self.dx_target_scheme)

    def dx_codes_history_extractor(self, dx_codes, admission_ids):
        for subject_id, subject_admission_ids in admission_ids.items():
            subject_admission_ids = sorted(subject_admission_ids)
            history = set()
            vec = jnp.zeros(len(self.dx_target_scheme))
            for adm_id in subject_admission_ids:
                history.update(dx_codes[adm_id].codeset)
                vec = jnp.maximum(vec, dx_codes[adm_id].vec)
                yield adm_id, Codes(history.copy(), vec, self.dx_target_scheme)

    def outcome_extractor(self, dx_codes):
        for adm_id, _dx_codes in dx_codes.items():
            outcome_codes = self.outcome_scheme.map_codeset(
                _dx_codes.codeset, self.dx_target_scheme)
            outcome_vec = self.outcome_scheme.codeset2vec(
                outcome_codes, self.dx_target_scheme)

            yield adm_id, Codes(outcome_codes, outcome_vec,
                                self.outcome_scheme)

    def procedure_extractor(self):
        c_adm_id = self.colname["int_proc"]["admission_id"]
        c_code = self.colname["int_proc"]["code"]
        c_start_time = self.colname["int_proc"]["start_time"]
        c_end_time = self.colname["int_proc"]["end_time"]
        scheme = self.int_proc_source_scheme
        c_admittime = self.colname["adm"]["admittime"]
        c_dischtime = self.colname["adm"]["dischtime"]

        df = self.df["int_proc"].merge(self.df["adm"],
                                       on=c_adm_id,
                                       how='inner')
        df[c_start_time] = (df[c_start_time] -
                            df[c_admittime]).dt.total_seconds() / 3600.0
        df[c_end_time] = (df[c_end_time] -
                          df[c_admittime]).dt.total_seconds() / 3600.0
        dischtime = (self.df["adm"][c_dischtime] -
                     self.df["adm"][c_admittime]).dt.total_seconds() / 3600.0
        dischtime = dict(zip(self.df["adm"][c_adm_id], dischtime))

        agg_rep = AggregateRepresentation(self.int_proc_source_scheme,
                                          self.int_proc_target_scheme)
        for adm_id, proc_df in df.groupby(c_adm_id):
            index = []
            rate = []
            start_t = []
            end_t = []
            for idx, row in proc_df.iterrows():
                index.append(scheme.index[row[c_code]])
                rate.append(1.0)
                start_t.append(row[c_start_time])
                end_t.append(row[c_end_time])
            ii = InpatientInput(index=jnp.array(index),
                                rate=jnp.array(rate),
                                starttime=jnp.array(start_t),
                                endtime=jnp.array(end_t),
                                size=len(self.int_proc_source_scheme.index))
            yield adm_id, InpatientSegmentedInput.from_input(
                ii, agg_rep, start_time=0.0, end_time=dischtime[adm_id])

    def inputs_extractor(self):
        c_adm_id = self.colname["int_input"]["admission_id"]
        c_code = self.colname["int_input"]["code"]
        c_start_time = self.colname["int_input"]["start_time"]
        c_end_time = self.colname["int_input"]["end_time"]
        c_rate = self.colname["int_input"]["rate"]
        c_admittime = self.colname["adm"]["admittime"]

        scheme = self.int_input_source_scheme
        df = self.df['int_input'].merge(self.df['adm'],
                                        on=c_adm_id,
                                        how='inner')
        df[c_start_time] = (df[c_start_time] -
                            df[c_admittime]).dt.total_seconds() / 3600.0
        df[c_end_time] = (df[c_end_time] -
                          df[c_admittime]).dt.total_seconds() / 3600.0

        for adm_id, input_df in df.groupby(c_adm_id):
            index = []
            rate = []
            start_t = []
            end_t = []
            for idx, row in input_df.iterrows():
                index.append(scheme.index[row[c_code]])
                rate.append(row[c_rate])
                start_t.append(row[c_start_time])
                end_t.append(row[c_end_time])
            yield adm_id, InpatientInput(
                index=jnp.array(index),
                rate=jnp.array(rate),
                starttime=jnp.array(start_t),
                endtime=jnp.array(end_t),
                size=len(self.int_input_source_scheme.index))

    def observables_extractor(self):
        c_adm_id = self.colname["obs"]["admission_id"]
        c_code = self.colname["obs"]["code"]
        c_time = self.colname["obs"]["timestamp"]
        c_value = self.colname["obs"]["value"]
        scheme = self.obs_scheme
        df = self.df["obs"].merge(self.df["adm"], on=c_adm_id, how='inner')
        c_admittime = self.colname["adm"]["admittime"]
        df[c_time] = (df[c_time] - df[c_admittime]).dt.total_seconds() / 3600.0
        df['code_index'] = df[c_code].apply(lambda c: scheme.index[c])
        for adm_id, obs_df in df.groupby(c_adm_id):
            times = []
            values = []
            masks = []

            for time, time_obs_df in obs_df.groupby(c_time):
                times.append(time)
                idx = time_obs_df['code_index'].values
                val = time_obs_df[c_value].values
                v = jnp.zeros(len(scheme.index))
                values.append(v.at[idx].set(val))
                masks.append(v.at[idx].set(1.0))

            times = jnp.array(times)
            sorter = jnp.argsort(times)
            values = jnp.vstack(values)[sorter]
            masks = jnp.vstack(masks)[sorter]

            yield adm_id, InpatientObservables(time=times,
                                               value=values,
                                               mask=masks)


def load_dataset(label):

    if label == 'M3':
        return MIMIC3EHRDataset.from_meta_json(f'{_META_DIR}/mimic3_meta.json')
    if label == 'M4':
        return MIMIC4EHRDataset.from_meta_json(f'{_META_DIR}/mimic4_meta.json')
    if label == 'CPRD':
        return CPRDEHRDataset.from_meta_json(f'{_META_DIR}/cprd_meta.json')
    if label == 'M4ICU':
        return MIMIC4ICUDataset.from_meta_json(
            f'{_META_DIR}/mimic4icu_meta.json')


if __name__ == '__main__':
    logging.root.level = logging.DEBUG
    m4inpatient_dataset = load_dataset('M4ICU')
