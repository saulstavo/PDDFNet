for len in 96 192 336 720


do
  python -u PDDFNet.py \
  --root_path data/ETT-small \
  --data ETTm1 \
  --data_path ETTm1.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 192 \
  --groups 6 \
  --depth 2 \
  --patch_size 8 \
  --IPSDB True \
  --IPFEB False \
  --kernel_size 7 \
  --dropout 0.1 \
  --lr 0.0003 \
  --train_epochs 60 \
  --batch_size 512 \
  --patience 2 \
  --factor 0.1 \
  --train_epochs 60
done
