source /mnt/afs/zhengmingkai/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/finn/xuehuiwen/envs/csq

cd /mnt/afs/zhengmingkai/whl/llamagen/tokenizer/tokenizer_image
export PYTHONPATH=/mnt/afs/zhengmingkai/whl/llamagen
export TORCH_HOME=/mnt/afs/zhengmingkai/.cache/torch

torchrun --nproc_per_node=8 bsqdc_train.py --sample

chmod -R 777 /mnt/afs/zhengmingkai/whl/llamagen/




torchrun --nproc_per_node=1 --master_port=29510 bsqdcs_train.py --sample



 ssh -L 2222:106.75.235.249:5000 mac-bridge-shared

 cd /mnt/afs/liyu1/0312/qworld_2-1_training_copy/qworld_2-1_training_copy/_my_configs/2511/11214_z10_diffsynth_ace_4b_qw_ds_v0_9_w8_tos_z1_nbucketing_60M_v3/