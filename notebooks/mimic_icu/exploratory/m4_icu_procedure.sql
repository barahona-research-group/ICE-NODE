select pi.hadm_id
, mimiciv_derived.DATETIME_DIFF(pi.chartdate, a.admittime, 'DAY') AS offset_days
, pi.icd_code
, pi.icd_version
, di.long_title
FROM mimiciv_hosp.procedures_icd pi
INNER JOIN mimiciv_hosp.d_icd_procedures di
  ON pi.icd_version = di.icd_version
  AND pi.icd_code = di.icd_code
INNER JOIN mimiciv_hosp.admissions a
  ON pi.hadm_id = a.hadm_id