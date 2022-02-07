drop table if exists discarded_study_trials;
select trial_id
into temporary table discarded_study_trials
from trials where state = 'FAIL';

delete from trial_values tv where tv.trial_id in (select trial_id from discarded_study_trials);
delete from trial_intermediate_values tiv where tiv.trial_id in (select trial_id from discarded_study_trials);
delete from trial_params tp where tp.trial_id in (select trial_id from discarded_study_trials);
delete from trial_system_attributes tsa where trial_id in (select trial_id from discarded_study_trials);
delete from trial_user_attributes tua where trial_id in (select trial_id from discarded_study_trials);
delete from trials where trial_id in (select trial_id from discarded_study_trials);
