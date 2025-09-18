for len in 96 192 336 720


do
  python -u PDDFNet.py \
  --root_path data/traffic \
  --data custom \
  --data_path traffic.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 512 \
  --groups 15 \
  --depth 2 \
  --patch_size 48 \
  --IPSDB False \
  --IPFEB False \
  --kernel_size 3 \
  --dropout 0.1 \
  --lr 0.0003 \
  --train_epochs 60 \
  --batch_size 4 \
  --factor 0.5 \
  --patience 2 \
  --early_stop_patience 10
done
