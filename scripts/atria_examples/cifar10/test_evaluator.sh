#!/bin/bash -l

# Start from the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Search for the project root by looking for a marker file or directory (e.g., .git)
find_project_root() {
    dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" || -f "$dir/pyproject.toml" || -f "$dir/setup.py" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_ROOT=$(find_project_root "$SCRIPT_DIR")
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Could not find project root."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/../atria_core/src/"

python \
    $SCRIPT_DIR/../../../src/atria/task_pipelines/evaluator.py \
    dataset@data_pipeline.dataset=cifar10/default \
    data_pipeline.dataset.data_dir=/tmp/cifar10/ \
    model_pipeline.model.model_name=resnet50 \
    $@