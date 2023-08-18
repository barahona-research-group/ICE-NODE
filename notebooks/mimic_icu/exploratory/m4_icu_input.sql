SELECT
    a.hadm_id
    , mimiciv_derived.DATETIME_DIFF(pe.starttime, a.admittime, 'MINUTE') AS start_offset
    , mimiciv_derived.DATETIME_DIFF(pe.endtime, a.admittime, 'MINUTE') AS end_offset
    , di.label
    , pe.value
FROM mimiciv_hosp.admissions a
INNER JOIN mimiciv_icu.icustays i
    ON a.hadm_id = i.hadm_id
LEFT JOIN mimiciv_icu.procedureevents pe
    ON i.stay_id = pe.stay_id
LEFT JOIN mimiciv_icu.d_items di
    ON pe.itemid = di.itemid