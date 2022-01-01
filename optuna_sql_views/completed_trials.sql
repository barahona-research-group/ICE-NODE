select t.trial_id, s.study_name, tv.value, tiv.best_value,  ps.parameters_size
from trial_values as tv
left join trials as t on tv.trial_id = t.trial_id
left join studies as s on s.study_id = t.study_id
left join
(select tiv.trial_id,  max(tiv.intermediate_value) as best_value
            from trial_intermediate_values tiv group by tiv.trial_id) as tiv on tiv.trial_id = t.trial_id
left join (select trial_id, cast(value_json as int) / 1000 as parameters_size from trial_user_attributes where key= 'parameters_size') as ps on t.trial_id = ps.trial_id
order by tv.value desc