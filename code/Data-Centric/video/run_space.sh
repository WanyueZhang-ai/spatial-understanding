# 22GB
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
VIDEO_MAX_PIXELS=200704 \
FPS_MAX_FRAMES=32 \
swift sft \
    --model YOUR_PATH \
    --dataset YOUR_PATH \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-6 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 1000 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --max_length 8000 \
    --output_dir YOUR_PATH \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --lr_scheduler_type "cosine" \
    --save_only_model \
    --model_type qwen2_5_vl
