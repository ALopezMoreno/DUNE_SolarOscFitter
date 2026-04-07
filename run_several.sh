#!/bin/bash

for i in {1..10}
do
  julia -t 10 src/readConfig.jl config.yaml
done
