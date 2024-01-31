from lib.ehr._dataset_mimic4 import (AdmissionMixedICDSQLTableConfig, StaticSQLTableConfig, AdmissionSQLTableConfig,
                                     AdmissionTimestampedMultiColumnSQLTableConfig,
                                     AdmissionTimestampedCodedValueSQLTableConfig,
                                     RatedInputSQLTableConfig)

ADMISSIONS_CONF = AdmissionSQLTableConfig(name="admissions",
                                          query=(r"""
select hadm_id as {admission_id_alias}, 
subject_id as {subject_id_alias},
admittime as {admission_time_alias},
dischtime as {discharge_time_alias}
from mimiciv_hosp.admissions 
"""))

STATIC_CONF = StaticSQLTableConfig(name="static",
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

DX_DISCHARGE_CONF = AdmissionMixedICDSQLTableConfig(name="dx_discharge",
                                                    query=(r"""
select hadm_id as {admission_id_alias}, 
        icd_code as {icd_code_alias}, 
        icd_version as {icd_version_alias}
from mimiciv_hosp.diagnoses_icd 
"""),
                                                    space_query=(r"""
select icd_code as {icd_code_alias},
        icd_version as {icd_version_alias},
        long_title as {description_alias}
from mimiciv_hosp.d_icd_diagnoses
"""))

RENAL_OUT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_out",
                                                               attributes=['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
                                                               query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.kdigo_uo as uo
inner join mimiciv_icu.icustays icu
on icu.stay_id = uo.stay_id
    """))

RENAL_CREAT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_creat",
                                                                 attributes=['creat'],
                                                                 query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.kdigo_creatinine
    """))

RENAL_AKI_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                               attributes=['aki_stage_smoothed', 'aki_binary'],
                                                               query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from (select hadm_id, charttime, aki_stage_smoothed, 
    case when aki_stage_smoothed = 1 then 1 else 0 end as aki_binary from mimiciv_derived.kdigo_stages)
    """))

SOFA_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                          attributes=["sofa_24hours"],
                                                          query=(r"""
select hadm_id {admission_id_alias}, {attributes}, s.endtime {time_alias} 
from mimiciv_derived.sofa as s
inner join mimiciv_icu.icustays icu on s.stay_id = icu.stay_id    
"""))

BLOOD_GAS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_gas",
                                                               attributes=['so2', 'po2', 'pco2', 'fio2',
                                                                           'fio2_chartevents',
                                                                           'aado2', 'aado2_calc',
                                                                           'pao2fio2ratio', 'ph',
                                                                           'baseexcess', 'bicarbonate', 'totalco2',
                                                                           'hematocrit',
                                                                           'hemoglobin',
                                                                           'carboxyhemoglobin', 'methemoglobin',
                                                                           'chloride', 'calcium', 'temperature',
                                                                           'potassium',
                                                                           'sodium', 'lactate', 'glucose'],
                                                               query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.bg as bg
where hadm_id is not null
"""))

BLOOD_CHEMISTRY_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_chemistry",
                                                                     attributes=['albumin', 'globulin', 'total_protein',
                                                                                 'aniongap', 'bicarbonate', 'bun',
                                                                                 'calcium', 'chloride',
                                                                                 'creatinine', 'glucose', 'sodium',
                                                                                 'potassium'],
                                                                     query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.chemistry as ch
where hadm_id is not null
"""))

CARDIAC_MARKER_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cardiac_marker",
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

WEIGHT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="weight",
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

CBC_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cbc",
                                                         attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv',
                                                                     'platelet',
                                                                     'rbc', 'rdw', 'wbc'],
                                                         query=(r"""
select hadm_id {admission_id_alias}, {attributes}, cbc.charttime {time_alias}
from mimiciv_derived.complete_blood_count as cbc
where hadm_id is not null
"""))

VITAL_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="vital",
                                                           attributes=['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni',
                                                                       'dbp_ni',
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
GCS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="gcs",
                                                         attributes=['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes',
                                                                     'gcs_unable'],
                                                         query=(r"""
select hadm_id {admission_id_alias}, {attributes}, gcs.charttime {time_alias}
from mimiciv_derived.gcs as gcs
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = gcs.stay_id
"""))

OBS_TABLE_CONFIG = AdmissionTimestampedCodedValueSQLTableConfig(components=[
    RENAL_OUT_CONF,
    RENAL_CREAT_CONF,
    RENAL_AKI_CONF,
    SOFA_CONF,
    BLOOD_GAS_CONF,
    BLOOD_CHEMISTRY_CONF,
    CARDIAC_MARKER_CONF,
    WEIGHT_CONF,
    CBC_CONF,
    VITAL_CONF,
    GCS_CONF
])

## Inputs - Canonicalise

ICU_INPUT_CONF = RatedInputSQLTableConfig(name="int_input",
                                          query=(r"""
select
    a.hadm_id as {admission_id_alias}
    , inp.itemid as {code_alias}
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
select di.itemid as {code_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.inputevents ie
    on di.itemid = ie.itemid
where di.itemid is not null
group by di.itemid
"""))

## Procedures - Canonicalise and Refine
ICU_PROC_CONF = IntervalICUProcedureSQLTableConfig(name="int_proc_icu",
                                                   query=(r"""
select a.hadm_id as {admission_id_alias}
    , pe.itemid as {code_alias}
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
select di.itemid as {code_alias}, max(di.label) as {description_alias}
from  mimiciv_icu.d_items di
inner join mimiciv_icu.procedureevents pe
    on di.itemid = pe.itemid
where di.itemid is not null
group by di.itemid
"""))

HOSP_PROC_CONF = IntervalBasedMixedICDSQLTableConfig(name="int_proc_icd",
                                                     query=(r"""
select pi.hadm_id as {admission_id_alias}
, (pi.chartdate)::timestamp as {start_time_alias}
, (pi.chartdate + interval '1 hour')::timestamp as {end_time_alias}
, pi.icd_code as {icd_code_alias}
, pi.icd_version as {icd_version_alias}
, di.long_title as {description_alias}
FROM mimiciv_hosp.procedures_icd pi
INNER JOIN mimiciv_hosp.d_icd_procedures di
  ON pi.icd_version = di.icd_version
  AND pi.icd_code = di.icd_code
INNER JOIN mimiciv_hosp.admissions a
  ON pi.hadm_id = a.hadm_id
"""),
                                                     space_query=(r"""

select di.icd_code  as {icd_code_alias},  
di.icd_version as {icd_version_alias},
max(di.long_title) as {description_alias}
from mimiciv_hosp.d_icd_procedures di
where di.icd_code is not null and di.icd_version is not null
group by di.icd_code, di.icd_version
"""))
