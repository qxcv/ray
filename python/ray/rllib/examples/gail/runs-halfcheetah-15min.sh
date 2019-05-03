#!/bin/bash

# Useful script for doing small benchmarks that show TD3 learning thread
# throughput anomaly. A few minutes minutes ought to be enough to shake out most
# issues with initialisation conditions; we only measure stats at the end.
python gail.py with halfcheetah-full-nogpu-one-worker.yaml td3_conf.pure_exploration_steps=0 td3_conf.learning_starts=0 max_time_s=900
python gail.py with halfcheetah-full-nogpu.yaml td3_conf.pure_exploration_steps=0 td3_conf.learning_starts=0 max_time_s=900
python gail.py with halfcheetah-full.yaml td3_conf.pure_exploration_steps=0 td3_conf.learning_starts=0 max_time_s=900  # <-- don't need GPU worker to spot the bug, but will compute anyway
