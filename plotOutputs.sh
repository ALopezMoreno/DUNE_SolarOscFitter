#!/bin/bash

# Directory containing the files
DIRECTORY="./outputs/"

# Iterate over each file starting with "both_" in the specified directory
for file in "$DIRECTORY"/both_*; do
    # Check if the file exists to avoid errors if no files match
    if [ -e "$file" ]; then
        # Run the Python script with the full path of the file
        python3 utils/plotChainOutput.py "$file"
    else
        echo "No files starting with 'ES_' found in the directory."
    fi
done
