WITH trop AS
(
    SELECT specimen_id, MAX(valuenum) AS troponin_t
    FROM mimiciv_hosp.labevents
    WHERE itemid = 51003
    GROUP BY specimen_id
)
SELECT
    a.hadm_id
    , mimiciv_derived.DATETIME_DIFF(c.charttime, a.admittime, 'MINUTE') AS offset
    , trop.troponin_t
    , c.ntprobnp
    , c.ck_mb
FROM mimiciv_hosp.admissions a
LEFT JOIN mimiciv_derived.cardiac_marker c
  ON a.hadm_id = c.hadm_id
LEFT JOIN trop
  ON c.specimen_id = trop.specimen_id