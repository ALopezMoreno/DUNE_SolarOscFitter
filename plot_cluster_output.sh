#!/bin/bash

# 1. Build the argument list
files=()
for i in {1..25}; do
    files+=("cluster_output/CC-onlly__sergioReco_5MeV__water100cmBG__4%8B-sys__2%BG-sys__dm-7.5_$i")
done

# 2. Run Python with all files as arguments
python3 utils/plotOutput.py -e  -o images/long_water100_5MeV_7.5 "${files[@]}"
