drop table if exists discarded_studies;
--    create temporary table discarded_studies
--   (
--         study_id integer
--    );
--    insert into discarded_studies
--    values
--    (23832),
--    (24191) ;
create temporary table discarded_studies as
    select study_id from studies where study_id =31721;


drop table if exists discarded_study_trials;
select trial_id
into temporary table discarded_study_trials
from trials where study_id in (select study_id from discarded_studies);

delete from trial_values tv where tv.trial_id in (select trial_id from discarded_study_trials);
delete from trial_intermediate_values tiv where tiv.trial_id in (select trial_id from discarded_study_trials);
delete from trial_params tp where tp.trial_id in (select trial_id from discarded_study_trials);
delete from trial_system_attributes tsa where trial_id in (select trial_id from discarded_study_trials);
delete from trial_user_attributes tua where trial_id in (select trial_id from discarded_study_trials);
delete from trials where trial_id in (select trial_id from discarded_study_trials);
delete from study_directions sd where study_id in (select study_id from discarded_studies);
delete from study_user_attributes sua where study_id in (select study_id from discarded_studies);
delete from study_system_attributes ssa where study_id in (select study_id from discarded_studies);
delete from studies where study_id in (select study_id from discarded_studies);

