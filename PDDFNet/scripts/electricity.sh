for len in 96 192 336 720

do
  python -u PDDFNet.py \
  --root_path data/electricity \
  --data custom \
  --data_path electricity.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 448 \
  --groups 19 \
  --depth 1 \
  --patch_size 32 \
  --IPSDB False \
  --IPFEB False \
  --kernel_size 7 \
  --dropout 0.1 \
  --lr 0.0002 \
  --train_epochs 60 \
  --batch_size 8 \
  --factor 0.5 \
  --patience 2 \
  --early_stop_patience 6
done
