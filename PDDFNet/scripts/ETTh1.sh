for len in 96 192 336 720

do
  python -u PDDFNet.py \
  --root_path data/ETT-small \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len $len \
  --emb_dim 256 \
  --groups 1 \
  --depth 1 \
  --patch_size 32 \
  --kernel_size 9 \
  --dropout 0.1 \
  --lr 0.0007 \
  --batch_size 512 \
  --patience 2 \
  --factor 0.1 \
  --IPSDB False \
  --IPFEB False \
  --train_epochs 60
done
