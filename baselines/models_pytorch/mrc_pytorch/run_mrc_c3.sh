#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=chinese_roberta_wwm_large_ext_pytorch
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=$OUTPUT_DIR/prev_trained_models/$MODEL_NAME
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="c3"

python run_c3.py \
  --gpu_ids="0" \
  --num_train_epochs=8 \
  --train_batch_size=1 \
  --eval_batch_size=24 \
  --gradient_accumulation_steps=16 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.05 \
  --max_seq_length=256 \
  --do_eval \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/pytorch_model.bin \
  --data_dir=$GLUE_DIR/$TASK_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  # --do_train \
