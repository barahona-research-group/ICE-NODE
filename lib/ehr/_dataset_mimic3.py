"""."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, ClassVar

import dask.dataframe as dd
import pandas as pd

from .concepts import (Patient, Admission, DemographicVectorConfig, StaticInfo)
from .dataset import Dataset, DatasetScheme


class MIMIC3Dataset(Dataset):
    df: Dict[str, dd.DataFrame]
    scheme: DatasetScheme
    seconds_scaler: ClassVar[float] = 1 / 3600.0  # convert seconds to hours

    def to_subjects(self, subject_ids: List[str], num_workers: int,
                    demographic_vector_config: DemographicVectorConfig,
                    target_scheme: DatasetScheme, **kwargs):

        subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
            subject_ids, target_scheme)
        admission_ids = self.adm_extractor(subject_ids)
        adm_ids_list = sum(map(list, admission_ids.values()), [])
        logging.debug('Extracting dx_discharge codes...')
        dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
        logging.debug('[DONE] Extracting dx_discharge codes')
        logging.debug('Extracting dx_discharge codes history...')
        dx_codes_history = dict(
            self.dx_codes_history_extractor(dx_codes, admission_ids,
                                            target_scheme))
        logging.debug('[DONE] Extracting dx_discharge codes history')
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
        c_adm_id = self.colname["dx_discharge"].admission_id
        c_code = self.colname["dx_discharge"].code

        df = self.df["dx_discharge"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        codes_df = {
            adm_id: codes_df
            for adm_id, codes_df in df.groupby(c_adm_id)
        }
        empty_vector = target_scheme.dx_discharge.empty_vector()
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
            vec = target_scheme.dx_discharge.empty_vector()
            yield (_adm_ids[0], vec)

            for prev_adm_id, adm_id in zip(_adm_ids[:-1], _adm_ids[1:]):
                if prev_adm_id in dx_codes:
                    vec = vec.union(dx_codes[prev_adm_id])
                yield (adm_id, vec)

    def outcome_extractor(self, dx_codes, target_scheme):
        return zip(dx_codes.keys(),
                   map(target_scheme.outcome.mapcodevector, dx_codes.values()))
