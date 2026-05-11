#!/bin/bash

for x in 1 2 3 4; do
    echo "Running config$x.yaml"
    julia -t 10 src/readConfig.jl "configs/config$x.yaml"
done
