#!/bin/bash
# Forward cli args to TPU VMs.

echo "#!/bin/bash
cd ~/jax-benchmarking
source ~/.bashrc
source env/bin/activate
" > _launch_run_pod.sh 

export |grep WANDB_SWEEP >> _launch_run_pod.sh 
export |grep WANDB_RUN >> _launch_run_pod.sh 

echo "python3 run_clm_flax.py --hostname tpu-pod-slice-vm-${TPU_NAME} $@" >> _launch_run_pod.sh

gcloud compute tpus tpu-vm ssh \
    --worker=all \
    --command "bash ~/jax-benchmarking/_launch_run_pod.sh" \
    ${TPU_NAME}