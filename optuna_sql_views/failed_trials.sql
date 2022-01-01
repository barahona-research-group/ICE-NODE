select t.trial_id, tua.nan, tua2.job_id from trials as t
left join (select tua.trial_id, tua.value_json as nan from trial_user_attributes as tua where tua.key = 'nan') as tua
on t.trial_id = tua.trial_id
left join (select tua2.trial_id, tua2.value_json as job_id from trial_user_attributes as tua2  where tua2.key = 'job_id') as tua2
on t.trial_id = tua2.trial_id
where t.state = 'FAIL' and nan is null
