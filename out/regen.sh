#!/bin/bash
cd /home/user/DoomFly
for pair in "0 1" "2 3" "4 5"; do set -- $pair; for sd in $1 $2; do (timeout 1700 python3 src/run_episode.py --duration 50 --tag v2_s$sd --seed $sd --no-spikes > out/v2_s$sd.log 2>&1; python3 src/analyze_episode.py v2_s$sd > out/analyze_v2_s$sd.out 2>&1) & done; wait; done
(timeout 900 python3 src/run_episode.py --duration 30 --tag v2_side_real --no-spikes --scripted L:15,R:15 --teacher-gap 1.2 2.0 > out/v2_side_real.log 2>&1) &
(timeout 900 python3 src/run_episode.py --duration 30 --tag v2_side_shuf --no-spikes --scripted L:15,R:15 --teacher-gap 1.2 2.0 --shuffle 0 > out/v2_side_shuf.log 2>&1) &
wait
python3 out/side_eval.py v2_side_real v2_side_shuf > out/v2_side_eval.out 2>&1
timeout 1700 python3 src/run_episode.py --duration 50 --tag v2_shuffled --shuffle 0 --no-spikes > out/v2_shuffled.log 2>&1; python3 src/analyze_episode.py v2_shuffled > out/analyze_v2_shuffled.out 2>&1
echo REGEN_DONE > out/regen.done
