#!/bin/bash
# The synthetic experiment: every method of synthetic/run.py in order, each over the instance sets it runs on (grid,
# sweep, two-structure cell, scale cell, delta cell), one process at a time, with the machine kept awake (caffeinate,
# macOS) while the script runs. A method skips what already has a line, so starting the script again resumes where it
# stopped. Each method's output goes to results/logs/<method>.log. results/logs/protocol.log gets the commit the
# script runs from (git describe, with -dirty when tracked files have uncommitted changes; untracked files do not
# count) and each method's start and end with its exit code. A method that fails is logged so and the next one starts.
# Python's linear algebra runs on one thread, as MATLAB and Gurobi do. MATLAB is the R2026a path below unless the
# environment's MATLAB names another.
#
# Start:  nohup bash run_synthetic.sh > /dev/null 2>&1 &
# Stop:   pkill -f "run_synthetic.sh|maxcon_pruning.synthetic.run"; pkill -f "astar_jobs|influence_jobs"
cd "$(dirname "$0")" || exit 1
export MATLAB=${MATLAB:-/Applications/MATLAB_R2026a.app/bin/matlab}
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
LOG=results/logs/protocol.log
if pgrep -f "maxcon_pruning\.(synthetic|real)\.|astar_jobs|influence_jobs" > /dev/null; then
    echo "$(date '+%F %T') not started: another run or MATLAB job is active" | tee -a "$LOG"
    exit 1
fi
caffeinate -is -w $$ &                                   # keeps the machine awake until this script exits
echo "$(date '+%F %T') commit $(git describe --tags --always --dirty 2>/dev/null || echo unknown)" >> "$LOG"

for method in generate prune trim gore wi-mbf astar-survivors astar-trimmed astar-gore astar-alone; do
    echo "$(date '+%F %T') start $method" >> "$LOG"
    .venv/bin/python -B -m maxcon_pruning.synthetic.run "$method" >> "results/logs/$method.log" 2>&1
    code=$?                                              # before any other command resets $?
    echo "$(date '+%F %T') end $method (exit $code)" >> "$LOG"
done
echo "$(date '+%F %T') protocol done" >> "$LOG"
