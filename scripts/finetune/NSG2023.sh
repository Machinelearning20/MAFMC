device=1

# finetune
# python -u run.py \
#     --is_training 1 \
#     --load_pretrain 0 \
#     --transfer_data NSG2023 \
#     --transfer_root_path dataset/NSG2023/ \
#     --transfer_data_path NSG2023.csv \
#     --data NSG2023 \
#     --root_path dataset/NSG2023/ \
#     --data_path NSG2023.csv \
#     --dropout 0.1 \
#     --learning_rate 0.0001 \
#     --gpu $device


for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data NSG2023 \
    --transfer_root_path dataset/NSG2023/ \
    --transfer_data_path NSG2023.csv \
    --data NSG2023 \
    --root_path dataset/NSG2023/ \
    --data_path NSG2023.csv \
    --ar_pred_len $pred \
    --gpu $device
  done