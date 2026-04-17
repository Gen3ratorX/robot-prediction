#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run_ros2_experiment.sh RUN_NAME DURATION_SEC [options]

Example:
  ./scripts/run_ros2_experiment.sh approaching_run_01 20 \
    --expected-movement approaching \
    --notes "Direct frontal approach in lab lighting"

Options passed after DURATION_SEC are forwarded to ros2_experiment_logger.
EOF
  exit 1
fi

RUN_NAME="$1"
DURATION="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"

mkdir -p "${RESULTS_DIR}"

if [[ -f "${HOME}/robot_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/robot_env/bin/activate"
fi

if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi

if [[ -f "${HOME}/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/ros2_ws/install/setup.bash"
fi

export PYTHONPATH="${HOME}/robot_env/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo "Logging ROS2 experiment to ${RUN_DIR}"

ros2 run robot_prediction ros2_experiment_logger \
  --output-dir "${RESULTS_DIR}" \
  --run-name "${RUN_NAME}" \
  --duration "${DURATION}" \
  "$@"

python3 -m robot_prediction.plot_experiment_results \
  --run-dir "${RUN_DIR}"

echo "Finished. Outputs:"
echo "  ${RUN_DIR}/events.csv"
echo "  ${RUN_DIR}/summary.csv"
echo "  ${RUN_DIR}/summary_table.csv"
echo "  ${RUN_DIR}/metrics.json"
