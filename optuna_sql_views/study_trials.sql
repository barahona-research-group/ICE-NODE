SELECT min(s.study_name) as study, count(case  when t.state='RUNNING' then 1 end) as running, count(case when t.state='COMPLETE' then 1 end) as completed, count(case when t.state='PRUNED' then 1 end) as pruned, count(case when t.state='FAIL' then 1 end) as fail, count(t.trial_id) as total from trials as t
left join studies as s on t.study_id = s.study_id
group by t.study_id
union
(select 'total' as study, count(case  when t.state='RUNNING' then 1 end) as running, count(case when t.state='COMPLETE' then 1 end) as completed, count(case when t.state='PRUNED' then 1 end) as pruned, count(case when t.state='FAIL' then 1 end) as fail, count(t.trial_id) as total from trials as t)
order by study asc