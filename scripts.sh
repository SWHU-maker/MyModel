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
batch_sizes="64 128"
dropouts="0.1"

for bs in $batch_sizes
do
  for drop in $dropouts
  do
    for pred_len in $pred_lens
    do
      echo "Running with pred_len=$pred_len, batch_size=$bs, dropout=$drop"
      
      python -u run.py \
        --random_seed $random_seed \
        --is_training 1 \
        --batch_size $bs \
        --dropout $drop \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'_drop'$drop \
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
