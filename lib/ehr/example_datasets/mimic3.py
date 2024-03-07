"""."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, ClassVar

import dask.dataframe as dd
import pandas as pd

from lib.ehr.tvx_concepts import (Patient, Admission, DemographicVectorConfig, StaticInfo)
from lib.ehr.dataset import Dataset, DatasetScheme


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

