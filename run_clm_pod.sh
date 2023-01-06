#!/bin/bash
cd ~/jax-benchmarking
source ~/.bashrc
source env/bin/activate

python3 run_clm_flax.py $@