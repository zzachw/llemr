#!/bin/bash

export PYTHONPATH=$(pwd)

script_dir=$(dirname "$0")

python $script_dir/preprocess/precompute_one_event_embeddings.py
