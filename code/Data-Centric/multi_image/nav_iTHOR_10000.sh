CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=16384 \
swift sft \
    --model YOUR_PATH \
    --dataset 'YOUR_PATH' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --split_dataset_ratio 0.1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir YOUR_PATH \
    --dataloader_num_workers 4 \
    --dataset_num_proc 64 \
    --attn_impl flash_attn \
    --lazy_tokenize true \
    --deepspeed zero2 \