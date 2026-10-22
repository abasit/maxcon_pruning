#!/bin/bash
# The real-data experiment in three stages, one process at a time, with the machine kept awake (caffeinate, macOS)
# while the script runs:
#   1  the main and limitation sets (MAIN): every method of real/run.py in the order of its METHODS, one start,
#      WI and MBF-MaxCon at REPS repetitions
#   2  the summary set (SUMMARY): prune (one start), trim, gore, wi-mbf at SUMMARY_REPS repetitions, astar-survivors,
#      astar-trimmed and astar-alone (on instances.ALONE)
#   3  the repeat stage on MAIN: the pruning's starts 2 to STARTS, start by start over every dataset (every instance
#      gets start j before any gets start j + 1, so the stage can be stopped at any time), each followed by the
#      trimming of the new LO-RANSAC lines
# A method skips what already has a line, so starting the script again resumes where it stopped. Each method's output
# goes to results/logs/real_<method>.log. results/logs/real_protocol.log gets the commit the script runs from (git
# describe, with -dirty when tracked files have uncommitted changes; untracked files do not count) and each stage's
# methods' start and end on each dataset with its exit code. A method that fails is logged so and the next one starts.
# Python's linear algebra runs on one thread, as MATLAB and Gurobi do. MATLAB is the R2026a path below unless the
# environment's MATLAB names another. MAIN, SUMMARY, STARTS, REPS and SUMMARY_REPS can be set in the environment (the
# defaults are the recorded run's values).
#
# Start:  nohup bash run_real.sh > /dev/null 2>&1 &
# Stop:   pkill -f "run_real.sh|maxcon_pruning.real.run"; pkill -f "astar_jobs|influence_jobs"
cd "$(dirname "$0")" || exit 1
MAIN=${MAIN:-"homogr evd vgg adelaide"}
SUMMARY=${SUMMARY:-"hpatches"}
STARTS=${STARTS:-10} REPS=${REPS:-10} SUMMARY_REPS=${SUMMARY_REPS:-5}
export MATLAB=${MATLAB:-/Applications/MATLAB_R2026a.app/bin/matlab}
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
LOG=results/logs/real_protocol.log
if pgrep -f "maxcon_pruning\.(synthetic|real)\.|astar_jobs|influence_jobs" > /dev/null; then
    echo "$(date '+%F %T') not started: another run or MATLAB job is active" | tee -a "$LOG"
    exit 1
fi
caffeinate -is -w $$ &                                   # keeps the machine awake until this script exits
echo "$(date '+%F %T') commit $(git describe --tags --always --dirty 2>/dev/null || echo unknown); main $MAIN," \
    "summary $SUMMARY, starts $STARTS, repetitions $REPS and $SUMMARY_REPS" >> "$LOG"

run() {                                                  # run STAGE METHOD DATASET [ARGS...]
    local stage=$1 method=$2 dataset=$3
    shift 3
    echo "$(date '+%F %T') start $stage $method $dataset $*" >> "$LOG"
    .venv/bin/python -B -m maxcon_pruning.real.run "$method" "$dataset" "$@" >> "results/logs/real_$method.log" 2>&1
    local code=$?                                        # before any other command resets $?
    echo "$(date '+%F %T') end $stage $method $dataset (exit $code)" >> "$LOG"
}

for method in prune trim gore gore-tests gore-cap wi-mbf astar-survivors astar-trimmed astar-gore astar-alone; do
    for dataset in $MAIN; do
        case $method in
            prune) run stage1 prune "$dataset" --repeats 1 ;;
            wi-mbf) run stage1 wi-mbf "$dataset" --repeats "$REPS" ;;
            *) run stage1 "$method" "$dataset" ;;
        esac
    done
done
for method in prune trim gore wi-mbf astar-survivors astar-trimmed astar-alone; do
    for dataset in $SUMMARY; do
        case $method in
            prune) run stage2 prune "$dataset" --repeats 1 ;;
            wi-mbf) run stage2 wi-mbf "$dataset" --repeats "$SUMMARY_REPS" ;;
            *) run stage2 "$method" "$dataset" ;;
        esac
    done
done
for (( start = 2; start <= STARTS; start++ )); do
    for dataset in $MAIN; do
        run "stage3 start $start" prune "$dataset" --repeats "$start"
        run "stage3 start $start" trim "$dataset"
    done
done
echo "$(date '+%F %T') protocol done" >> "$LOG"
