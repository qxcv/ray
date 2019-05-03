#!/bin/bash

# Hacky script to run HalfCheetah with a bunch of configs. I used this to make
# my poster plots.

for iter in {1..4}; do
    echo "Running iteration ${iter}"
    python gail.py with halfcheetah-full.yaml
    python gail.py with halfcheetah-full-nogpu-one-worker.yaml
    python gail.py with halfcheetah-full-nogpu.yaml
    python gail.py with halfcheetah-sync.yaml
done
