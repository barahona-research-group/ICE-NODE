SELECT
      pat.subject_id
    , adm.hadm_id
    , icu.stay_id
    , ROW_NUMBER() OVER (PARTITION BY pat.subject_id ORDER BY icu.intime) AS icu_stay_num
    , DENSE_RANK() OVER (PARTITION BY pat.subject_id ORDER BY adm.admittime) AS hosp_stay_num
    , CASE
        WHEN FIRST_VALUE(icu.stay_id) OVER icustay_window = icu.stay_id THEN 1
        ELSE 0
      END AS pat_count
    , pat.anchor_age + (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year) AS age
    , pat.gender
    , adm.insurance
    , icu.first_careunit
    , icu.los AS icu_los
    , mimiciv_derived.DATETIME_DIFF(adm.dischtime, adm.admittime, 'HOUR') / 24 AS hosp_los
    , pat.dod
    , mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') AS days_to_death
    -- mortality flags
    , CASE WHEN mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') = 0 THEN 1 ELSE 0 END AS hospital_mortality
    , CASE WHEN mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(icu.outtime AS DATE), 'DAY') = 0 THEN 1 ELSE 0 END AS icu_mortality
FROM mimiciv_hosp.patients pat
INNER JOIN mimiciv_hosp.admissions adm
    ON pat.subject_id = adm.subject_id
INNER JOIN mimiciv_icu.icustays icu
    ON adm.hadm_id = icu.hadm_id
WINDOW hadm_window AS (PARTITION BY pat.subject_id ORDER BY adm.admittime)
     , icustay_window AS (PARTITION BY pat.subject_id ORDER BY icu.intime)