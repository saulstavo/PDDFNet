for len in 96 192 336 720


do
  python -u PDDFNet.py \
  --root_path data/exchange_rate \
  --data custom \
  --data_path exchange_rate.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 128 \
  --groups 1 \
  --depth 4 \
  --patch_size 8 \
  --IPSDB True \
  --IPFEB True \
  --kernel_size 7 \
  --dropout 0.1 \
  --lr 0.0002 \
  --train_epochs 60 \
  --batch_size 128 \
  --patience 1 \
  --factor 0.5
done
