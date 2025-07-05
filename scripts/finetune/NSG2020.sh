device=1

# finetune
# python -u run.py \
#     --is_training 1 \
#     --load_pretrain 0 \
#     --transfer_data NSG2020 \
#     --transfer_root_path dataset/NSG2020/ \
#     --transfer_data_path NSG2020.csv \
#     --data NSG2020 \
#     --root_path dataset/NSG2020/ \
#     --data_path NSG2020.csv \
#     --dropout 0.1 \
#     --learning_rate 0.0001 \
#     --gpu $device


for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data NSG2020 \
    --transfer_root_path dataset/NSG2020/ \
    --transfer_data_path NSG2020.csv \
    --data NSG2020 \
    --root_path dataset/NSG2020/ \
    --data_path NSG2020.csv \
    --ar_pred_len $pred \
    --gpu $device
  done