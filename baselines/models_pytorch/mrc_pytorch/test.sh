CURRENT_DIR=/lustre/home/acct-seexy/seexy/main/qsf/nlp/mrc_pytorch
export EXP_NAME=out_layer_4 
export MODEL_NAME=chinese_roberta_wwm_large_ext_pytorch
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=$OUTPUT_DIR/prev_trained_models/$MODEL_NAME
export GLUE_DIR=$CURRENT_DIR/mrc_data
TASK_NAME="c3"

python run_c3_4.py \
  --gpu_ids="0,1" \
  --num_train_epochs=8 \
  --train_batch_size=16 \
  --eval_batch_size=24 \
  --gradient_accumulation_steps=4 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.05 \
  --max_seq_length=512 \
  --do_eval \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/pytorch_model.bin \
  --data_dir=$GLUE_DIR/$TASK_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/$EXP_NAME \
  --do_train \