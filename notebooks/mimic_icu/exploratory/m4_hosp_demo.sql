SELECT
      pat.subject_id
    , adm.hadm_id
    , DENSE_RANK() OVER hadm_window AS hosp_stay_num
    , CASE
        WHEN FIRST_VALUE(adm.hadm_id) OVER hadm_window = adm.hadm_id THEN 1
        ELSE 0
      END AS pat_count
    , pat.anchor_age + (EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year) AS age
    , pat.gender
    , adm.insurance
    , mimiciv_derived.DATETIME_DIFF(adm.dischtime, adm.admittime, 'HOUR') / 24 AS hosp_los
    , pat.dod
    , mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') AS days_to_death
    -- mortality flags
    , CASE WHEN mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') = 0 THEN 1 ELSE 0 END AS hospital_mortality
FROM mimiciv_hosp.patients pat
INNER JOIN mimiciv_hosp.admissions adm
    ON pat.subject_id = adm.subject_id
WINDOW hadm_window AS (PARTITION BY pat.subject_id ORDER BY adm.admittime)