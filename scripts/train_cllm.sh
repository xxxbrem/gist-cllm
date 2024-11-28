export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=consistency_llm

model_path=$1
trajectory_file=$2
output_path=$3
n_token_seq_size=$4
use_cross_entropy=$5
qlora=$6

torchrun -m --nnodes=1 --nproc_per_node=4 --rdzv_id=101 --rdzv_endpoint='localhost:6666' \
    --master_port 10000 \
    gist_cllm.train_cllm_global \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --use_cross_entropy ${use_cross_entropy} \
    --qlora ${qlora} 
