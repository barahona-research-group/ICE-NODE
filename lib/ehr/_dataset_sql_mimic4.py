from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import field
from functools import cached_property
from typing import Union, Dict, List, Optional, Tuple

import pandas as pd
import sqlalchemy
from sqlalchemy import Engine

from ._coding_scheme_icd import ICDScheme
from .coding_scheme import (OutcomeExtractor, CodingScheme, CodingSchemeConfig, FlatScheme, CodeMap)
from .dataset import (DatasetScheme, DatasetSchemeConfig)
from ..base import Config, Module


class MIMIC4DatasetScheme(DatasetScheme):
    dx: Union[Dict[str, CodingScheme], CodingScheme]

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        if isinstance(self.config.dx, dict):
            self.dx = {
                version: CodingScheme.from_name(scheme)
                for version, scheme in self.config.dx.items()
            }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        attrs = list(k for k in source.scheme_dict if k != 'dx')
        for attr in attrs:
            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"
        for version, s_scheme in source.dx.items():
            t_scheme = target.dx
            assert s_scheme.mapper_to(
                t_scheme), f"Cannot map dx (version={version}) \
                from {s_scheme} to {t_scheme}"

    def dx_mapper(self, target_scheme: DatasetScheme):
        return {
            version: s_dx.mapper_to(target_scheme.dx.name)
            for version, s_dx in self.dx.items()
        }

    @property
    def supported_target_scheme_options(self):
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__,) +
               getattr(self, k).supported_targets
            for k in self.scheme_dict
        }
        supported_dx_targets = {
            version: (scheme.__class__.__name__,) + scheme.supported_targets
            for version, scheme in self.dx.items()
        }
        supproted_attr_targets['dx'] = list(
            set.intersection(*map(set, supported_dx_targets.values())))
        supported_outcomes = {
            version: OutcomeExtractor.supported_outcomes(scheme)
            for version, scheme in self.dx.items()
        }
        supproted_attr_targets['outcome'] = list(
            set.intersection(*map(set, supported_outcomes.values())))

        return supproted_attr_targets


class MIMIC4ICUDatasetSchemeConfig(DatasetSchemeConfig):
    int_proc: str = 'int_mimic4_proc'  # -> 'int_mimic4_grouped_proc'
    int_input: str = 'int_mimic4_input'  # -> 'int_mimic4_input_group'
    obs: str = 'mimic4_obs'


class MIMIC4ICUDatasetScheme(MIMIC4DatasetScheme):
    int_proc: CodingScheme
    int_input: CodingScheme
    obs: CodingScheme

    def make_target_scheme_config(self, **kwargs):
        assert 'outcome' in kwargs, "Outcome must be specified"
        return self.config.update(int_proc='int_mimic4_grouped_proc',
                                  int_input='int_mimic4_input_group',
                                  **kwargs)


class MIMICIVSQLTableConfig(Config):
    name: str
    query: str
    admission_id_alias: str = 'hadm_id'

    @property
    def alias_dict(self):
        return {k: v for k, v in self.as_dict().items() if k.endswith('_alias')}

    @property
    def alias_id_dict(self):
        return {k: v for k, v in self.alias_dict.items() if '_id_' in k}


class AdmissionLinkedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    admission_id_alias: str = 'hadm_id'


class SubjectLinkedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    subject_id_alias: str = 'subject_id'


class TimestampedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    attributes: Tuple[str] = tuple()
    time_alias: str = 'time_bin'


class CategoricalMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    space_query: str = ''
    description_alias: str = 'description'


class IntervalMIMICIVSQLTableConfig(CategoricalMIMICIVSQLTableConfig):
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'


class AdmissionMIMICIVSQLTableConfig(AdmissionLinkedMIMICIVSQLTableConfig, SubjectLinkedMIMICIVSQLTableConfig):
    admission_time_alias: str = 'admittime'
    discharge_time_alias: str = 'dischtime'


class StaticMIMICIVSQLTableConfig(SubjectLinkedMIMICIVSQLTableConfig):
    gender_alias: str = 'gender'
    anchor_year_alias: str = 'anchor_year'
    anchor_age_alias: str = 'anchor_age'
    race_alias: str = 'race'
    date_of_birth: str = 'dob'


class MixedICDMIMICIVSQLTableConfig(CategoricalMIMICIVSQLTableConfig):
    code_alias: str = 'icd_code'
    version_alias: str = 'icd_version'


class ItemBasedMIMICIVSQLTableConfig(CategoricalMIMICIVSQLTableConfig):
    item_id_alias: str = 'itemid'


class DxDischargeMIMICIVSQLTableConfig(AdmissionLinkedMIMICIVSQLTableConfig, MixedICDMIMICIVSQLTableConfig):
    pass


class IntervalHospProcedureMIMICIVSQLTableConfig(IntervalMIMICIVSQLTableConfig, MixedICDMIMICIVSQLTableConfig):
    pass


class IntervalICUProcedureMIMICIVSQLTableConfig(IntervalMIMICIVSQLTableConfig, ItemBasedMIMICIVSQLTableConfig):
    pass


class RatedInputMIMICIVSQLTableConfig(IntervalMIMICIVSQLTableConfig, ItemBasedMIMICIVSQLTableConfig):
    rate_alias: str = 'rate'
    rate_unit_alias: str = 'rateuom'
    amount_alias: str = 'amount'
    amount_unit_alias: str = 'amountuom'


admissions_conf = AdmissionMIMICIVSQLTableConfig(name="admissions",
                                                 query=(r"""
select hadm_id as {admission_id_alias}, 
subject_id as {subject_id_alias},
admittime as {admission_time_alias},
dischtime as {discharge_time_alias}
from mimiciv_hosp.admissions 
"""))

static_conf = StaticMIMICIVSQLTableConfig(name="static",
                                          query=(r"""
select 
p.subject_id as {subject_id_alias},
p.gender as {gender_alias},
a.race as {race_alias},
p.anchor_age as {anchor_age_alias},
p.anchor_year as {anchor_year_alias}
from mimiciv_hosp.patients p
left join 
(select subject_id, max(race) as race
from mimiciv_hosp.admissions
group by subject_id) as a
on p.subject_id = a.subject_id
"""))

dx_discharge_conf = DxDischargeMIMICIVSQLTableConfig(name="dx",
                                                     query=(r"""
select hadm_id as {admission_id_alias}, 
        icd_code as {code_alias}, 
        icd_version as {version_alias}
from mimiciv_hosp.diagnoses_icd 
"""),
                                                     space_query=(r"""
select icd_code as {code_alias},
        icd_version as {version_alias},
        long_title as {description_alias}
from mimiciv_hosp.d_icd_diagnoses
"""))

renal_out_conf = TimestampedMIMICIVSQLTableConfig(name="renal_out",
                                                  attributes=['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
                                                  query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.kdigo_uo as uo
inner join mimiciv_icu.icustays icu
on icu.stay_id = uo.stay_id
    """))

renal_creat_conf = TimestampedMIMICIVSQLTableConfig(name="renal_creat",
                                                    attributes=['creat'],
                                                    query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.kdigo_creatinine
    """))

renal_aki_conf = TimestampedMIMICIVSQLTableConfig(name="renal_aki",
                                                  attributes=['aki_stage_smoothed', 'aki_binary'],
                                                  query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from (select hadm_id, charttime, aki_stage_smoothed, 
    case when aki_stage_smoothed = 1 then 1 else 0 end as aki_binary from mimiciv_derived.kdigo_stages)
    """))

sofa_conf = TimestampedMIMICIVSQLTableConfig(name="renal_aki",
                                             attributes=["sofa_24hours"],
                                             query=(r"""
select hadm_id {admission_id_alias}, {attributes}, s.endtime {time_alias} 
from mimiciv_derived.sofa as s
inner join mimiciv_icu.icustays icu on s.stay_id = icu.stay_id    
"""))

blood_gas_conf = TimestampedMIMICIVSQLTableConfig(name="blood_gas",
                                                  attributes=['so2', 'po2', 'pco2', 'fio2', 'fio2_chartevents',
                                                              'aado2', 'aado2_calc',
                                                              'pao2fio2ratio', 'ph',
                                                              'baseexcess', 'bicarbonate', 'totalco2', 'hematocrit',
                                                              'hemoglobin',
                                                              'carboxyhemoglobin', 'methemoglobin',
                                                              'chloride', 'calcium', 'temperature', 'potassium',
                                                              'sodium', 'lactate', 'glucose'],
                                                  query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.bg as bg
where hadm_id is not null
"""))

blood_chemistry_conf = TimestampedMIMICIVSQLTableConfig(name="blood_chemistry",
                                                        attributes=['albumin', 'globulin', 'total_protein',
                                                                    'aniongap', 'bicarbonate', 'bun',
                                                                    'calcium', 'chloride',
                                                                    'creatinine', 'glucose', 'sodium', 'potassium'],
                                                        query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.chemistry as ch
where hadm_id is not null
"""))

cardiac_marker_conf = TimestampedMIMICIVSQLTableConfig(name="cardiac_marker",
                                                       attributes=['troponin_t', 'ntprobnp', 'ck_mb'],
                                                       query=(r"""
with trop as
(
    select specimen_id, MAX(valuenum) as troponin_t
    from mimiciv_hosp.labevents
    where itemid = 51003
    group by specimen_id
)
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_hosp.admissions a
left join mimiciv_derived.cardiac_marker c
  on a.hadm_id = c.hadm_id
left join trop
  on c.specimen_id = trop.specimen_id
where c.hadm_id is not null
"""))

weight_conf = TimestampedMIMICIVSQLTableConfig(name="weight",
                                               attributes=['weight'],
                                               query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, w.time_bin {time_alias}
 from (
 (select stay_id, w.weight, w.starttime time_bin
  from mimiciv_derived.weight_durations as w)
 union all
 (select stay_id, w.weight, w.endtime time_bin
     from mimiciv_derived.weight_durations as w)
 ) w
inner join mimiciv_icu.icustays icu on w.stay_id = icu.stay_id
"""))

cbc_conf = TimestampedMIMICIVSQLTableConfig(name="cbc",
                                            attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'platelet',
                                                        'rbc', 'rdw', 'wbc'],
                                            query=(r"""
select hadm_id {admission_id_alias}, {attributes}, cbc.charttime {time_alias}
from mimiciv_derived.complete_blood_count as cbc
where hadm_id is not null
"""))

vital_conf = TimestampedMIMICIVSQLTableConfig(name="vital",
                                              attributes=['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni',
                                                          'mbp_ni', 'resp_rate',
                                                          'temperature', 'spo2',
                                                          'glucose'],
                                              query=(r"""
select hadm_id {admission_id_alias}, {attributes}, v.charttime {time_alias}
from mimiciv_derived.vitalsign as v
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = v.stay_id
"""))

# Glasgow Coma Scale, a measure of neurological function
gcs_conf = TimestampedMIMICIVSQLTableConfig(name="gcs",
                                            attributes=['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable'],
                                            query=(r"""
select hadm_id {admission_id_alias}, {attributes}, gcs.charttime {time_alias}
from mimiciv_derived.gcs as gcs
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = gcs.stay_id
"""))

## Inputs - Canonicalise

icu_input_conf = RatedInputMIMICIVSQLTableConfig(name="int_input",
                                                 query=(r"""
select
    a.hadm_id as {admission_id_alias}
    , inp.itemid as {item_id_alias}
    , inp.starttime as {start_time_alias}
    , inp.endtime as {end_time_alias}
    , di.label as {description_alias}
    , inp.rate  as {rate_alias}
    , inp.amount as {amount_alias}
    , inp.rateuom as {rate_unit_alias}
    , inp.amountuom as {amount_unit_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.inputevents inp
    on i.stay_id = inp.stay_id
left join mimiciv_icu.d_items di
    on inp.itemid = di.itemid
"""),
                                                 space_query=(r"""
select di.itemid as {item_id_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.inputevents ie
    on di.itemid = ie.itemid
where di.itemid is not null
group by di.itemid
"""))

## Procedures - Canonicalise and Refine
icuproc_conf = IntervalICUProcedureMIMICIVSQLTableConfig(name="int_proc_icu",
                                                         query=(r"""
select a.hadm_id as {admission_id_alias}
    , pe.itemid as {item_id_alias}
    , pe.starttime as {start_time_alias}
    , pe.endtime as {end_time_alias}
    , di.label as {description_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.procedureevents pe
    on i.stay_id = pe.stay_id
left join mimiciv_icu.d_items di
    on pe.itemid = di.itemid
"""),
                                                         space_query=(r"""
select di.itemid as {item_id_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.procedureevents pe
    on di.itemid = pe.itemid
where di.itemid is not null
group by di.itemid
"""))

hospicdproc_conf = IntervalHospProcedureMIMICIVSQLTableConfig(name="int_proc_icd",
                                                              query=(r"""
select pi.hadm_id as {admission_id_alias}
, (pi.chartdate)::timestamp as {start_time_alias}
, (pi.chartdate + interval '1 hour')::timestamp as {end_time_alias}
, pi.icd_code as {code_alias}
, pi.icd_version as {version_alias}
, di.long_title as {description_alias}
FROM mimiciv_hosp.procedures_icd pi
INNER JOIN mimiciv_hosp.d_icd_procedures di
  ON pi.icd_version = di.icd_version
  AND pi.icd_code = di.icd_code
INNER JOIN mimiciv_hosp.admissions a
  ON pi.hadm_id = a.hadm_id
"""),
                                                              space_query=(r"""

select di.icd_code  as {code_alias},  
di.icd_version as {version_alias},
max(di.long_title) as {description_alias}
from mimiciv_hosp.d_icd_procedures di
where di.icd_code is not null and di.icd_version is not null
group by di.icd_code, di.icd_version
"""))


class MIMICIVSQLTable(Module):
    config: MIMICIVSQLTableConfig

    def _coerce_id_to_str(self, df: pd.DataFrame):
        """
        Some of the integer ids in the database when downloaded are stored as floats.
        A fix is to coerce them to integers then fix as strings.
        """
        id_alias_dict = self.config.alias_id_dict
        # coerce to integers then fix as strings.
        int_dtypes = {k: int for k in id_alias_dict.values() if k in df.columns}
        str_dtypes = {k: str for k in id_alias_dict.values() if k in df.columns}
        return df.astype(int_dtypes).astype(str_dtypes)

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))


class TimestampedMIMICIVSQLTable(MIMICIVSQLTable):
    config: TimestampedMIMICIVSQLTableConfig

    def __call__(self, engine: Engine,
                 attributes: List[str]):
        assert len(set(attributes)) == len(attributes), \
            f"Duplicate attributes {attributes}"
        assert all(a in self.config.attributes for a in attributes), \
            f"Some attributes {attributes} not in {self.config.attributes}"
        query = self.config.query.format(attributes=','.join(attributes),
                                         **self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))

    @staticmethod
    def space(configs: List[TimestampedMIMICIVSQLTableConfig]):
        rows = []
        for table in configs:
            rows.extend([(table.name, a) for a in table.attributes])
        df = pd.DataFrame(rows, columns=['table_name', 'attribute']).set_index('table_name', drop=True)
        return df.sort_values('attribute').sort_index()


class StaticMIMICIVSQLTable(MIMICIVSQLTable):
    config: StaticMIMICIVSQLTableConfig

    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        df = self._coerce_id_to_str(pd.read_sql(query, engine,
                                                coerce_float=False))

        anchor_date = pd.to_datetime(df[self.config.anchor_year_alias],
                                     format='%Y').dt.normalize()
        anchor_age = df[self.config.anchor_age_alias].map(
            lambda y: pd.DateOffset(years=-y))
        df[self.config.date_of_birth] = anchor_date + anchor_age

        return df

    def gender_space(self):
        pass

    def ethnicity_space(self):
        pass

    def register_gender_scheme(self, name: str,
                               engine: Engine,
                               gender_selection: Optional[pd.DataFrame]):
        pass

    def register_ethnicity_scheme(self, name: str,
                                  engine: Engine,
                                  ethnicity_selection: Optional[pd.DataFrame]):
        pass


class CategoricalMIMICIVSQLTable(MIMICIVSQLTable):
    config: CategoricalMIMICIVSQLTableConfig

    def space(self, engine: Engine):
        query = self.config.space_query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine,
                                                  coerce_float=False))


class MixedICDMIMICIVSQLTable(CategoricalMIMICIVSQLTable):
    config: MixedICDMIMICIVSQLTableConfig

    def register_scheme(self, name: str,
                        engine: Engine,
                        icd_schemes: Dict[str, str],
                        icd_version_selection: Optional[pd.DataFrame]):

        c_version = self.config.version_alias
        c_code = self.config.code_alias
        c_desc = self.config.description_alias
        supported_space = self.space(engine)
        if icd_version_selection is None:
            icd_version_selection = supported_space[[c_version, c_code, c_desc]].drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
        else:
            if c_desc not in icd_version_selection.columns:
                icd_version_selection = pd.merge(icd_version_selection,
                                                 supported_space[[c_version, c_code, c_desc]],
                                                 on=[c_version, c_code], how='left')
            icd_version_selection = icd_version_selection[[c_version, c_code, c_desc]]
            icd_version_selection = icd_version_selection.drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
            # Check that all the ICD codes-versions are supported.
            for version, codes in icd_version_selection.groupby(c_version):
                support_subset = supported_space[supported_space[c_version] == version]
                assert codes[c_code].isin(support_subset[c_code]).all(), "Some ICD codes are not supported"
        scheme = MixedICDMIMICIVScheme.from_selection(name, icd_version_selection,
                                                      version_alias=c_version,
                                                      icd_code_alias=c_code,
                                                      description_alias=c_desc,
                                                      icd_schemes=icd_schemes)
        MixedICDMIMICIVScheme.register_scheme(scheme)
        scheme.register_maps_loaders()

        return scheme


class ItemBasedMIMICIVSQLTable(CategoricalMIMICIVSQLTable):
    config: ItemBasedMIMICIVSQLTableConfig

    @staticmethod
    def _register_flat_scheme(name: str, codes: List[str], desc: Dict[str, str]):
        codes = sorted(codes)
        index = dict(zip(codes, range(len(codes))))

        scheme = FlatScheme(CodingSchemeConfig(name),
                            codes=codes,
                            desc=desc,
                            index=index)

        FlatScheme.register_scheme(scheme)
        return scheme

    def register_scheme(self, name: str,
                        engine: Engine, itemid_selection: Optional[pd.DataFrame]):

        c_itemid = self.config.item_id_alias
        c_desc = self.config.description_alias
        supported_space = self.space(engine)
        if itemid_selection is None:
            itemid_selection = supported_space[c_itemid].drop_duplicates().astype(str).tolist()
        else:
            itemid_selection = itemid_selection[c_itemid].drop_duplicates().astype(str).tolist()

            assert len(set(itemid_selection) - set(supported_space[c_itemid])) == 0, \
                "Some item ids are not supported."
        desc = supported_space.set_index(c_itemid)[c_desc].to_dict()
        desc = {k: v for k, v in desc.items() if k in itemid_selection}
        return self._register_flat_scheme(name, itemid_selection, desc)


class ObservableMIMICScheme(FlatScheme):

    @classmethod
    def from_selection(cls, name: str, obs_variables: pd.DataFrame):
        """
        Create a scheme from a selection of observation variables.

        Args:
            name: Name of the scheme.
            obs_variables: A DataFrame containing the variables to include in the scheme.
                The DataFrame should have the following columns:
                    - table_name (index): The name of the table containing the variable.
                    - attribute: The name of the variable.

        Returns:
            (CodingScheme.FlatScheme) A new scheme containing the variables in obs_variables.
        """
        obs_variables = obs_variables.sort_values(['table_name', 'attribute'])
        # format codes to be of the form 'table_name.attribute'
        codes = (obs_variables.index + '.' + obs_variables['attribute']).tolist()
        desc = dict(zip(codes, codes))
        index = dict(zip(codes, range(len(codes))))

        return cls(CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc,
                   index=index)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'table_name', 'attribute']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split('.')) for c in self.codes],
                            columns=columns)


class MixedICDMIMICIVScheme(FlatScheme):
    _icd_schemes: Dict[str, ICDScheme]
    _sep: str = ':'

    def __init__(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str],
                 index: Dict[str, int], icd_schemes: Dict[str, ICDScheme], sep: str = ':'):
        super().__init__(config, codes=codes, desc=desc, index=index)
        self._icd_schemes = icd_schemes
        self._sep = sep

    @classmethod
    def from_selection(cls, name: str, icd_version_selection: pd.DataFrame,
                       version_alias: str, icd_code_alias: str, description_alias: str,
                       icd_schemes: Dict[str, str], sep: str = ':'):
        icd_version_selection = icd_version_selection.sort_values([version_alias, icd_code_alias])
        icd_version_selection = icd_version_selection.drop_duplicates([version_alias, icd_code_alias]).astype(str)
        assert icd_version_selection[version_alias].isin(icd_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_schemes))} are expected."

        # assert no duplicate (icd_code, icd_version)
        assert icd_version_selection.groupby([version_alias, icd_code_alias]).size().max() == 1, \
            "Duplicate (icd_code, icd_version) pairs are not allowed."

        icd_schemes_loaded: Dict[str, ICDScheme] = {k: ICDScheme.from_name(v) for k, v in icd_schemes.items()}

        assert all(isinstance(s, ICDScheme) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        for version, icd_df in icd_version_selection.groupby(version_alias):
            scheme = icd_schemes_loaded[version]
            icd_version_selection.loc[icd_df.index, icd_code_alias] = \
                icd_df[icd_code_alias].str.replace(' ', '').str.replace('.', '').map(scheme.add_dots)

        codes = (icd_version_selection['icd_version'] + sep + icd_version_selection['icd_code']).tolist()
        df = icd_version_selection.copy()
        df['code'] = codes
        index = dict(zip(codes, range(len(codes))))
        desc = df.set_index('code')[description_alias].to_dict()

        return MixedICDMIMICIVScheme(config=CodingSchemeConfig(name),
                                     codes=codes,
                                     desc=desc,
                                     index=index,
                                     icd_schemes=icd_schemes_loaded,
                                     sep=sep)

    def mixedcode_format_table(self, table: pd.DataFrame, code_alias: str, version_alias: str):
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        table = table.copy()
        assert version_alias in table.columns, f"Column {version_alias} not found."
        assert code_alias in table.columns, f"Column {code_alias} not found."
        assert table[version_alias].isin(self._icd_schemes).all(), \
            f"Only ICD version {list(self._icd_schemes.keys())} are expected."

        for version, icd_df in table.groupby(version_alias):
            scheme = self._icd_schemes[str(version)]
            table.loc[icd_df.index, code_alias] = icd_df[code_alias].str.replace(' ', '').str.replace('.', '').map(
                scheme.add_dots)

        # the version:icd_code format.
        table['code'] = table[version_alias] + self._sep + table[code_alias]

        # filter out codes that are not in the scheme.
        table = table[table['code'].isin(self.codes)].reset_index(drop=True)
        table = table.drop
        return table

    def generate_maps(self):
        """
        Register the mappings between the Mixed ICD scheme and the individual ICD scheme.
        For example, if the current `MixedICDMIMICIV` is mixing ICD-9 and ICD-10,
        then register the two mappings between this scheme and ICD-9 and ICD-10 separately.
        This assumes that the current runtime has already registered mappings
        between the individual ICD schemes.
        """

        # mixed2pure_maps has the form {icd_version: {mixed_code: {icd}}}.
        mixed2pure_maps = {}
        lost_codes = []
        dataframe = self.as_dataframe()
        for pure_version, pure_scheme in self._icd_schemes.items():
            # mixed2pure has the form {mixed_code: {icd}}.
            mixed2pure = defaultdict(set)
            for mixed_version, mixed_version_df in dataframe.groupby('icd_version'):
                icd2mixed = mixed_version_df.set_index('icd_code')['code'].to_dict()
                icd2mixed = {icd: mixed_code for icd, mixed_code in icd2mixed.items() if icd in pure_scheme.codes}
                assert len(icd2mixed) > 0, "No mapping between the mixed and pure ICD schemes was found."
                if mixed_version == pure_version:
                    mixed2pure.update({c: {icd} for icd, c in icd2mixed.items()})
                else:
                    # if mixed_version != pure_version, then retrieve
                    # the mapping between ICD-{mixed_version} and ICD-{pure_version}
                    pure_map = CodeMap.get_mapper(self._icd_schemes[mixed_version].name,
                                                  self._icd_schemes[pure_version].name)

                    for icd, mixed_code in icd2mixed.items():
                        if icd in pure_map:
                            mixed2pure[mixed_code].update(pure_map[icd])
                        else:
                            lost_codes.append(mixed_code)

            mixed2pure_maps[pure_version] = mixed2pure
        # register the mapping between the mixed and pure ICD schemes.
        for pure_version, mixed2pure in mixed2pure_maps.items():
            pure_scheme = self._icd_schemes[pure_version]
            conf = CodingSchemeConfig(self.name, pure_scheme.name)

            CodeMap.register_map(self.name,
                                 pure_scheme.name,
                                 CodeMap(conf, mixed2pure))

        lost_df = dataframe[dataframe['code'].isin(lost_codes)]
        if len(lost_df) > 0:
            logging.warning(f"Lost {len(lost_df)} codes when generating the mapping between the Mixed ICD"
                            "scheme and the individual ICD scheme.")
            logging.warning(lost_df.to_string().replace('\n', '\n\t'))

    def register_maps_loaders(self):
        """
        Register the lazy-loading of mapping between the Mixed ICD scheme and the individual ICD scheme.
        """
        for pure_version, pure_scheme in self._icd_schemes.items():
            CodeMap.register_map_loader(self.name, pure_scheme.name, self.generate_maps)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self._sep)) for c in self.codes],
                            columns=columns)


class MIMICIVSQLConfig(Config):
    host: str
    port: int
    user: str
    password: str
    dbname: str

    static_table: StaticMIMICIVSQLTableConfig = static_conf
    admissions_table: AdmissionMIMICIVSQLTableConfig = admissions_conf
    dx_discharge_table: DxDischargeMIMICIVSQLTableConfig = dx_discharge_conf
    obs_tables: List[TimestampedMIMICIVSQLTableConfig] = field(default_factory=lambda: [
        renal_out_conf,
        renal_creat_conf,
        renal_aki_conf,
        sofa_conf,
        blood_gas_conf,
        blood_chemistry_conf,
        cardiac_marker_conf,
        weight_conf,
        cbc_conf,
        vital_conf,
        gcs_conf
    ])
    icu_procedures_table: IntervalICUProcedureMIMICIVSQLTableConfig = icuproc_conf
    icu_inputs_table: RatedInputMIMICIVSQLTableConfig = icu_input_conf
    hosp_procedures_table: IntervalHospProcedureMIMICIVSQLTableConfig = hospicdproc_conf


class MIMICIVSQL(Module):
    config: MIMICIVSQLConfig

    def create_engine(self) -> Engine:
        return sqlalchemy.create_engine(
            f'postgresql+psycopg2://{self.config.user}:{self.config.password}@'
            f'{self.config.host}:{self.config.port}/{self.config.dbname}')

    def register_obs_scheme(self, name: str,
                            obs_variables: Optional[Union[pd.DataFrame, Dict[str, List[str]]]]):
        """
        From the given selection of observable variables `obs_variables`, generate a new scheme
        that can be used to generate for vectorisation of the timestamped observations.

        Args:
            name : The name of the scheme.
            obs_variables : A dictionary of table names and their corresponding attributes.
                If None, all supported variables will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """

        supported_obs_variables = self.supported_obs_variables
        if obs_variables is None:
            obs_variables = self.supported_obs_variables
        elif isinstance(obs_variables, dict):
            rows = []
            for table_name, attributes in obs_variables.items():
                assert table_name in supported_obs_variables.index, \
                    f"Table {table_name} not supported."
                assert all(a in supported_obs_variables.loc[table_name] for a in attributes), \
                    f"Some attributes {attributes} not in {supported_obs_variables.loc[table_name]}"
                rows.extend([(table_name, a) for a in attributes])
            obs_variables = pd.DataFrame(rows, columns=['table_name', 'attribute']).set_index('table_name', drop=True)

        obs_scheme = ObservableMIMICScheme.from_selection(name, obs_variables)
        FlatScheme.register_scheme(obs_scheme)
        return obs_scheme

    def register_icu_input_scheme(self, name: str, itemid_selection: Optional[pd.DataFrame]):
        """
        From the given selection of ICU input items `itemid_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU inputs. If `itemid_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            itemid_selection : A dataframe containing the `itemid`s to generate the new scheme. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = ItemBasedMIMICIVSQLTable(self.config.icu_inputs_table)
        return table.register_scheme(name, self.create_engine(), itemid_selection)

    def register_icu_procedure_scheme(self, name: str, itemid_selection: Optional[pd.DataFrame]):
        """
        From the given selection of ICU procedure items `itemid_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU procedures. If `itemid_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            itemid_selection : A dataframe containing the `itemid`s of choice. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = ItemBasedMIMICIVSQLTable(self.config.icu_procedures_table)
        return table.register_scheme(name, self.create_engine(), itemid_selection)

    def register_hosp_procedure_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
        """
        From the given selection of hospital procedure items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = MixedICDMIMICIVSQLTable(self.config.hosp_procedures_table)
        return table.register_scheme(name, self.create_engine(), {'9': 'pr_icd9', '10': 'pr_flat_icd10'},
                                     icd_version_selection)

    def register_dx_discharge_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
        """
        From the given selection of discharge diagnosis items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        table = MixedICDMIMICIVSQLTable(self.config.dx_discharge_table)
        return table.register_scheme(name, self.create_engine(), {'9': 'dx_icd9', '10': 'dx_flat_icd10'},
                                     icd_version_selection)

    @cached_property
    def supported_obs_variables(self) -> pd.DataFrame:
        return TimestampedMIMICIVSQLTable.space(self.config.obs_tables)

    @cached_property
    def supported_icu_procedures(self) -> pd.DataFrame:
        table = ItemBasedMIMICIVSQLTable(self.config.icu_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_inputs(self) -> pd.DataFrame:
        table = ItemBasedMIMICIVSQLTable(self.config.icu_inputs_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_hosp_procedures(self) -> pd.DataFrame:
        table = MixedICDMIMICIVSQLTable(self.config.hosp_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_dx_discharge(self) -> pd.DataFrame:
        table = MixedICDMIMICIVSQLTable(self.config.dx_discharge_table)
        return table.space(self.create_engine())

    def obs_table_interface(self, table_name: str) -> TimestampedMIMICIVSQLTable:
        table_conf = next(t for t in self.config.obs_tables if t.name == table_name)
        return TimestampedMIMICIVSQLTable(table_conf)

    def _extract_static_table(self, engine: Engine) -> pd.DataFrame:
        table = MIMICIVSQLTable(self.config.static_table)
        return table(engine)

    def _extract_admissions_table(self, engine: Engine) -> pd.DataFrame:
        table = MIMICIVSQLTable(self.config.admissions_table)
        return table(engine)

    def _extract_dx_discharge_table(self, engine: Engine, dx_discharge_scheme: MixedICDMIMICIVScheme) -> pd.DataFrame:
        table = MIMICIVSQLTable(self.config.dx_discharge_table)
        dataframe = table(engine)
        c_icd_version = self.config.dx_discharge_table.version_alias
        c_icd_code = self.config.dx_discharge_table.code_alias
        return dx_discharge_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def _extract_obs_table(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        dfs = dict()
        for table_name, attrs_df in obs_scheme.as_dataframe().groupby('table_name'):
            attributes = attrs_df['attribute'].tolist()
            attr2code = attrs_df.set_index('attribute')['code'].to_dict()
            # download the table.
            table = self.obs_table_interface(str(table_name))
            obs_df = table(engine, attributes)
            # melt the table. (admission_id, time, attribute, value)
            melted_obs_df = obs_df.melt(id_vars=[table.config.admission_id_alias, table.config.time_alias],
                                        var_name=['attribute'], value_name='value', value_vars=attributes)
            # add the code. (admission_id, time, attribute, value, code)
            melted_obs_df['code'] = melted_obs_df['attribute'].map(attr2code)
            melted_obs_df = melted_obs_df[melted_obs_df.value.notnull()]
            # drop the attribute. (admission_id, time, value, code)
            melted_obs_df = melted_obs_df.drop('attribute', axis=1)
            dfs[table_name] = melted_obs_df

        return pd.concat(dfs.values(), ignore_index=True)

    def _extract_icu_procedures_table(self, engine: Engine, icu_procedure_scheme: FlatScheme) -> pd.DataFrame:
        table = ItemBasedMIMICIVSQLTable(self.config.icu_procedures_table)
        c_itemid = self.config.icu_procedures_table.item_id_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_itemid].isin(icu_procedure_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_icu_inputs_table(self, engine: Engine, icu_input_scheme: FlatScheme) -> pd.DataFrame:
        table = ItemBasedMIMICIVSQLTable(self.config.icu_inputs_table)
        c_itemid = self.config.icu_inputs_table.item_id_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_itemid].isin(icu_input_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_hosp_procedures_table(self, engine: Engine,
                                       procedure_icd_scheme: MixedICDMIMICIVScheme) -> pd.DataFrame:
        table = MixedICDMIMICIVSQLTable(self.config.hosp_procedures_table)
        c_icd_code = self.config.hosp_procedures_table.code_alias
        c_icd_version = self.config.hosp_procedures_table.version_alias
        dataframe = table(engine)
        return procedure_icd_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def dataset_scheme_from_selection(self,
                                      gender_selection: Optional[pd.DataFrame] = None,
                                      ethnicity_selection: Optional[pd.DataFrame] = None,
                                      dx_discharge_selection: Optional[pd.DataFrame] = None,
                                      obs_selection: Optional[pd.DataFrame] = None,
                                      icu_input_selection: Optional[pd.DataFrame] = None,
                                      icu_procedure_selection: Optional[pd.DataFrame] = None,
                                      hosp_procedure_selection: Optional[pd.DataFrame] = None):
        """
        Create a dataset scheme from the given selection of variables.

        Args:

            gender_selection: A dataframe containing the `gender`s to generate the new scheme. If None, all supported items will be used.
            ethnicity_selection: A dataframe containing the `ethnicity`s to generate the new scheme. If None, all supported items will be used.
            dx_discharge_selection: A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.
            obs_selection: A dictionary of observation table names and their corresponding attributes.
                If None, all supported variables will be used.
            icu_input_selection: A dataframe containing the `itemid`s to generate the new scheme. If None, all supported items will be used.
            icu_procedure_selection: A dataframe containing the `itemid`s of choice. If None, all supported items will be used.
            hosp_procedure_selection: A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.DatasetScheme) A new scheme that is also registered in the current runtime.
        """
        pass

    def __call__(self, dataset_scheme: DatasetScheme) -> Dict[str, pd.DataFrame]:
        pass
