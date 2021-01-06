for BATCH in 64
do
  for LR in 0.001
  do
    for i in 104,104,8;
    do IFS=","; set -- $i;
      for ENCODE_LAYER in 5
      do
        for ATT_DROPOUT in 0.05
        do
          for RES_DROPOUT in 0.1
          do
            CUDA_VISIBLE_DEVICES=1 python models.py --patience=0 --dataset=snips --split=' ' --max_epochs=200 \
            --batch_size=${BATCH} --valid_data_path=test --lr=${LR} --alpha=0.6 --rm_nums=True --hidden_size=$1 \
            --filter_size=$2 --num_heads=$3 --encode_mode='UTF-8' --num_encoder_layers=${ENCODE_LAYER} \
            --attention_dropout=${ATT_DROPOUT} --residual_dropout=${RES_DROPOUT} --multiply_embedding_mode='none'
          done
        done
      done
    done
  done
done
