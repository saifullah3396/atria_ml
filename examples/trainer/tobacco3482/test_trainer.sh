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

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/../atria_core/src"

python \
    $PROJECT_ROOT/src/atria/task_pipelines/trainer.py \
    --config-name task_pipeline/trainer/image_classification \
    dataset@data_pipeline.dataset=classification/tobacco3482 \
    model_pipeline.model.model_name=resnet50 \
    data_pipeline.train_dataloader.batch_size=32 \
    data_pipeline.evaluation_dataloader.batch_size=32 \
    $@

# python \
    # $PROJECT_ROOT/src/atria/task_pipelines/trainer.py \
    # --config-name task_pipeline/trainer/sequence_classification \
    # dataset@data_pipeline.dataset=classification/tobacco3482 \
    # model_pipeline.model.model_name=microsoft/layoutlmv3-base \
    # data_pipeline.train_dataloader.batch_size=32 \
    # data_pipeline.evaluation_dataloader.batch_size=32 \
    # $@