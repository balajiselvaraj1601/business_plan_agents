#!/bin/bash

# Set debug flag - set to true for dry-run, false for actual sync
flag_debug=false

# Build the rclone command
cmd="rclone sync /Users/kzqr495/Documents/Agentic/business_plan_agents/ scp_se:/home/kzqr495/business_plan_agents/ \
  --exclude \".venv/**\" \
  --exclude \".git/**\" \
  --exclude \".tox/**\" \
  --exclude \".mypy_cache/**\" \
  --exclude \".ruff_cache/**\" \
  --exclude \".pytest_cache/**\" \
  --exclude \"__pycache__/**\" \
  --exclude \".coverage\" \
  --exclude \".DS_Store\" \
  --exclude \"*.pyc\" \
  --exclude \"*.pyo\" \
  --exclude \"src/business_plan_agents.egg-info/**\""

# Add dry-run flag if debug is enabled
if [ "$flag_debug" = true ]; then
    cmd="$cmd --dry-run"
fi

# Add progress flag
cmd="$cmd --progress"

# Execute the command
echo "Running: $cmd"
eval $cmd 