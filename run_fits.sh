#!/bin/bash

# Directory containing the config files
config_dir="configs"

# Check if the directory exists
if [ ! -d "$config_dir" ]; then
  echo "Directory $config_dir does not exist."
  exit 1
fi

# Iterate over each config_n.yaml file in the directory
for config_file in "$config_dir"/config_*.yaml; do
  # Check if the file exists (in case there are no matching files)
  if [ -e "$config_file" ]; then
    echo "Processing $config_file"
    # Run the Julia command with the current config file
    julia -t 10 src/readConfig.jl "$config_file"
  else
    echo "No config files found in $config_dir."
    exit 1
  fi
done
