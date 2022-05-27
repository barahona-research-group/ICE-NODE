select  t.trial_id, t.state, vals.value, invals.last_step, invals.best_value
from trial_values as vals
left join
     (
         select max(invals.step) as last_step, max(invals.intermediate_value) as best_value, invals.trial_id
         from trial_intermediate_values as invals
         group by invals.trial_id
     ) as invals
on vals.trial_id = invals.trial_id
inner join
     (
         select trial_id, state
         from trials
         where study_id = 8895
     ) as t
on vals.trial_id = t.trial_id
order by vals.value desc