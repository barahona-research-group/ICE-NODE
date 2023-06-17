select hadm_id,
       avg(so2) so2,
       date_trunc('hour', bg.charttime) hround_time
from mimiciv_derived.bg as bg
group by hadm_id, hround_time
