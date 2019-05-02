#!/bin/bash

# Hacky script to run HalfCheetah with a bunch of configs. I used this to make
# my poster plots.

python gail.py with halfcheetah-full.yaml
python gail.py with halfcheetah-full-nogpu-one-worker.yaml
python gail.py with halfcheetah-full-nogpu.yaml
