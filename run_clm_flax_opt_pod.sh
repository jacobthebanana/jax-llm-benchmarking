#!/bin/bash
cd ~/jax-benchmarking
source ~/.bashrc
source env/bin/activate

python run_clm_flax.py \
    --block_size=512 \
    --dataset_config_name=wikitext-103-v1 \
    --dataset_name=wikitext \
    --do_eval=1 --do_train=1 \
    --eval_steps=2500 \
    --logging_steps=1 \
    --model_name_or_path=facebook/opt-6.7b \
    --model_type=opt \
    --num_train_epochs=1 \
    --output_dir=/data/jax-benchmarking/models/sweep \
    --overwrite_output_dir=1 \
    --per_device_eval_batch_size=2 \
    --per_device_train_batch_size=2 \
    --save_steps=2500 \
    --wandb_entity=jacobthebanana \
    --wandb_project=jax-clm-benchmarking
