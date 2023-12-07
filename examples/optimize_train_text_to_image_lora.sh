#!/bin/bash

export MODEL_NAME=${MODEL_NAME:-"runwayml/stable-diffusion-v1-5"}
export OUTPUT_DIR=${OUTPUT_DIR:-"sddata/finetune/lora/pokemon"}
# export HUB_MODEL_ID=${HUB_MODEL_ID:-"pokemon-lora"}
export DATASET_NAME=${DATASET_NAME:-"lambdalabs/pokemon-blip-captions"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# python3 $SCRIPT_DIR/optimize_train_text_to_image_lora.py \
accelerate launch --mixed_precision="fp16" $SCRIPT_DIR/optimize_train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=7500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337 \
  --allow_tf32 \
  --sfast
  # --push_to_hub \
  # --hub_model_id=${HUB_MODEL_ID} \
  # --report_to=wandb \
