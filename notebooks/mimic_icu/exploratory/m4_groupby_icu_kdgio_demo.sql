SELECT
      MIN(pat.subject_id) AS subject_id
    , MIN(adm.hadm_id) AS hadm_id
    , icu.stay_id
    , icu.intime
    , DENSE_RANK() OVER subject_window AS hosp_stay_num
    , DENSE_RANK() OVER hosp_window AS icu_stay_num
    , MIN(kdigo.aki_stage_smoothed) as icu_min_aki
    , MAX(kdigo.aki_stage_smoothed) as icu_max_aki
    , CASE
        WHEN FIRST_VALUE(icu.stay_id) OVER subject_window = icu.stay_id THEN 1
        ELSE 0
      END AS pat_count
    , MIN(pat.anchor_age) + (EXTRACT(YEAR FROM MIN(adm.admittime)) - MIN(pat.anchor_year)) AS age
    , MIN(pat.gender) as gender
    , MIN(adm.insurance) as insurance
    , icu.los as icu_los
    , MIN(pat.dod) as dod
    -- mortality flags
    , CASE WHEN mimiciv_derived.DATETIME_DIFF(MIN(pat.dod), CAST(icu.outtime AS DATE), 'DAY') = 0 THEN 1 ELSE 0 END AS icu_mortality
FROM mimiciv_hosp.patients pat

INNER JOIN mimiciv_hosp.admissions adm
    ON pat.subject_id = adm.subject_id

INNER JOIN mimiciv_icu.icustays icu
    ON adm.hadm_id = icu.hadm_id

INNER JOIN mimiciv_derived.kdigo_stages kdigo
    ON icu.stay_id = kdigo.stay_id

GROUP BY icu.stay_id

WINDOW subject_window AS (PARTITION BY icu.subject_id ORDER BY icu.intime)
    , hosp_window AS (PARTITION BY icu.hadm_id ORDER BY icu.intime)

ORDER BY subject_id, hosp_stay_num, icu_stay_num