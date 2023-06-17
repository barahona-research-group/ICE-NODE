WITH trop AS
(
    SELECT specimen_id, MAX(valuenum) AS troponin_t
    FROM mimiciv_hosp.labevents
    WHERE itemid = 51003
    GROUP BY specimen_id
)
SELECT
    c.hadm_id
    , date_trunc('hour', c.charttime) hround_time
    , avg(trop.troponin_t) as troponin_t
    , avg(c.ntprobnp) as ntprobnp
    , avg(c.ck_mb) as ck_mb
FROM mimiciv_hosp.admissions a
LEFT JOIN mimiciv_derived.cardiac_marker c
  ON a.hadm_id = c.hadm_id
LEFT JOIN trop
  ON c.specimen_id = trop.specimen_id
GROUP BY c.hadm_id, hround_time