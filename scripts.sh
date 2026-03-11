export CUDA_VISIBLE_DEVICES=2
seq_len=96
model_name=PatchMamba
root_path_name=./datasets/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2024

# 超参数搜索空间
pred_lens="96 192 336 720"
batch_sizes="64"
dropouts="0 "
moe_weights="1"

for bs in $batch_sizes
do
  for drop in $dropouts
  do
    for moe_w in $moe_weights
    do
      for pred_len in $pred_lens
      do
        echo "Running with pred_len=$pred_len, batch_size=$bs, dropout=$drop, moe_loss_weight=$moe_w"
        
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --batch_size $bs \
          --dropout $drop \
          --moe_loss_weight $moe_w \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'_drop'$drop'_moe'$moe_w \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --gpu 0 \
        #   --use_multi_gpu \
        #   --devices '0,1,2,3' \
        
      done
    done
  done
done
