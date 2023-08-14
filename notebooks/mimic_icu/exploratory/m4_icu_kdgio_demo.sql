SELECT
      pat.subject_id
    , adm.hadm_id
    , icu.stay_id
    , DENSE_RANK() OVER subject_window AS hosp_stay_num
    , DENSE_RANK() OVER hosp_window AS icu_stay_num
    , DENSE_RANK() OVER icu_window AS aki_timestamp_index
    , MIN(aki_stage) OVER subject_window as current_subject_min_aki
    , MAX(aki_stage) OVER subject_window as current_subject_max_aki
    , MIN(aki_stage) OVER hosp_window as current_hosp_min_aki
    , MAX(aki_stage) OVER hosp_window as current_hosp_max_aki
    , MIN(aki_stage) OVER icu_window as current_icu_min_aki
    , MAX(aki_stage) OVER icu_window as current_icu_max_aki
    , agg_kdigo_subject.min_aki as subject_min_aki
    , agg_kdigo_subject.max_aki as subject_max_aki
    , agg_kdigo_hosp.min_aki as hosp_min_aki
    , agg_kdigo_hosp.max_aki as hosp_max_aki
    , agg_kdigo_icu.min_aki as icu_min_aki
    , agg_kdigo_icu.max_aki as icu_max_aki
    , kdigo.aki_stage_smoothed AS aki_stage
    , kdigo.charttime AS aki_time
    , CASE
        WHEN FIRST_VALUE(adm.hadm_id) OVER subject_window = adm.hadm_id THEN 1
        ELSE 0
      END AS pat_count
    , pat.anchor_age + (EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year) AS age
    , pat.gender
    , adm.insurance
    , mimiciv_derived.DATETIME_DIFF(adm.dischtime, adm.admittime, 'HOUR') / 24 AS hosp_los
    , icu.los as icu_los
    , pat.dod
    , mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') AS days_to_death
    -- mortality flags
    , CASE WHEN mimiciv_derived.DATETIME_DIFF(pat.dod, CAST(adm.dischtime AS DATE), 'DAY') = 0 THEN 1 ELSE 0 END AS hospital_mortality
FROM mimiciv_hosp.patients pat

INNER JOIN mimiciv_hosp.admissions adm
    ON pat.subject_id = adm.subject_id

INNER JOIN mimiciv_icu.icustays icu
    ON adm.hadm_id = icu.hadm_id

INNER JOIN mimiciv_derived.kdigo_stages kdigo
    ON icu.stay_id = kdigo.stay_id

INNER JOIN (SELECT MIN(aki_stage_smoothed) AS min_aki
                , MAX(aki_stage_smoothed) AS max_aki
                , stay_id
           FROM mimiciv_derived.kdigo_stages kdigo GROUP BY kdigo.stay_id) agg_kdigo_icu
    ON icu.stay_id = agg_kdigo_icu.stay_id

INNER JOIN (SELECT MIN(aki_stage_smoothed) AS min_aki
                , MAX(aki_stage_smoothed) AS max_aki
                , hadm_id
           FROM mimiciv_derived.kdigo_stages kdigo GROUP BY kdigo.hadm_id) agg_kdigo_hosp
    ON adm.hadm_id = agg_kdigo_hosp.hadm_id

INNER JOIN (SELECT MIN(aki_stage_smoothed) AS min_aki
                , MAX(aki_stage_smoothed) AS max_aki
                , subject_id
           FROM mimiciv_derived.kdigo_stages kdigo GROUP BY kdigo.subject_id) agg_kdigo_subject
    ON pat.subject_id = agg_kdigo_subject.subject_id

WINDOW subject_window AS (PARTITION BY pat.subject_id ORDER BY adm.admittime)
    , hosp_window AS (PARTITION BY adm.hadm_id ORDER BY icu.intime)
    , icu_window AS (PARTITION BY icu.stay_id ORDER BY kdigo.charttime )

ORDER BY subject_id, hosp_stay_num, icu_stay_num, aki_timestamp_index