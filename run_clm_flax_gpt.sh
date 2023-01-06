python run_clm_flax.py \
    --output_dir="/data/jax-benchmarking/models/norwegian-gpt2" \
    --model_type="gpt2" \
    --config_name="/data/jax-benchmarking/models/norwegian-gpt2" \
    --tokenizer_name="/data/jax-benchmarking/models/norwegian-gpt2" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
    --wandb_entity="jacobthebanana" \
    --wandb_project="jax-clm-benchmarking"