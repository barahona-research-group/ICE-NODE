-- Show best achieved intermediate results among all trials over all experiments.
select t.trial_id, t.state, tiv.best_value, s.study_name, p.progress, ps.parameters_size
from trials as t
join (select tiv.trial_id,  max(tiv.intermediate_value) as best_value
            from trial_intermediate_values tiv group by tiv.trial_id) as tiv on tiv.trial_id = t.trial_id
left join studies s on t.study_id = s.study_id
left join (select trial_id, cast(value_json as float4) as progress  from trial_user_attributes where key = 'progress') as p on t.trial_id = p.trial_id
left join (select trial_id, cast(value_json as int) / 1000 as parameters_size from trial_user_attributes where key= 'parameters_size') as ps on t.trial_id = ps.trial_id
order by tiv.best_value desc