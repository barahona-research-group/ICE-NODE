SELECT
    a.hadm_id
    , mimiciv_derived.DATETIME_DIFF(p.ordertime, a.admittime, 'DAY') AS offset
    , p.poe_id
    , p.order_type, p.order_subtype
    , p.transaction_type
    , pd.field_name
    , pd.field_value
FROM mimiciv_hosp.admissions a
INNER JOIN mimiciv_hosp.poe p
    ON a.hadm_id = p.hadm_id
LEFT JOIN  mimiciv_hosp.poe_detail pd
    ON p.poe_id = pd.poe_id