for len in 96 192 336 720

do
  python -u PDDFNet.py \
  --root_path data/ETT-small \
  --data ETTh2 \
  --data_path ETTh2.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 256 \
  --groups 4 \
  --depth 1 \
  --patch_size 8 \
  --IPSDB False \
  --IPFEB True \
  --kernel_size 7 \
  --dropout 0.1 \
  --lr 0.0003 \
  --train_epochs 60 \
  --batch_size 512 \
  --patience 2 \
  --factor 0.1
done
