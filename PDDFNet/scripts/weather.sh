for len in 96 192 336 720

do
  python -u PDDFNet.py \
  --root_path data/weather \
  --data custom \
  --data_path weather.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 160 \
  --groups 19 \
  --depth 1 \
  --patch_size 8 \
  --IPSDB True \
  --IPFEB False \
  --kernel_size 7 \
  --dropout 0.1 \
  --lr 0.0004 \
  --train_epochs 60 \
  --batch_size 512
done
