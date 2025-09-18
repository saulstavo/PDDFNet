for len in 96 192 336 720


do
  python -u PDDFNet.py \
  --root_path data/ETT-small \
  --data ETTm2 \
  --data_path ETTm2.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 64 \
  --groups 6 \
  --depth 2 \
  --patch_size 16 \
  --IPSDB True \
  --IPFEB False \
  --kernel_size 9 \
  --dropout 0.1 \
  --lr 0.0001 \
  --train_epochs 60 \
  --batch_size 512 \
  --patience 2 \
  --factor 0.1 \
  --train_epochs 60
done
