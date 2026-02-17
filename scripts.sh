
seq_len=96
model_name=PatchMambaRouter
root_path_name=./datasets/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2024

for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --batch_size 64 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21
done


