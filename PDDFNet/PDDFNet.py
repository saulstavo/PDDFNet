import argparse
import datetime
import os
import random

import lightning as L
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool
import torch.nn.functional as F

from lightning.pytorch.callbacks import EarlyStopping


class IntraPatchFrequencyEnhancement(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        freq_dim = feature_dim // 2 + 1

        # init
        self.complex_weight = nn.Parameter(
            torch.randn(freq_dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x_in):
        device = x_in.device
        B, N, C = x_in.shape

        x = x_in
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight.to(device))
        x_weighted = x_fft * weight

        x = torch.fft.irfft(x_weighted, n=C, dim=2, norm='ortho')
        return x


class moving_avg_enhanced(nn.Module):

    def __init__(self, kernel_size, dynamic_padding=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dynamic_padding = dynamic_padding
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        in: (batch*var, num_patches, embed_dim)
        out: (batch*var, num_patches, embed_dim)
        """
        orig_dim = x.shape[-1]

        if orig_dim < self.kernel_size:
            effective_ks = max(3, orig_dim // 2)
        else:
            effective_ks = self.kernel_size

        # padding
        if self.dynamic_padding:
            pad_left = (effective_ks - 1) // 2
            pad_right = effective_ks - 1 - pad_left
            padding = (pad_left, pad_right)
        else:
            padding = (0, 0)

        x = x.permute(0, 2, 1)  # [batch*var, embed_dim, num_patches]
        x = F.pad(x, padding, mode='replicate')
        moving_mean = self.avg(x)
        moving_mean = moving_mean.permute(0, 2, 1)
        return moving_mean


class InterPatchSeriesDecomposition(nn.Module):

    def __init__(self, kernel_size=5, adaptive=True):
        super().__init__()
        self.adaptive = adaptive
        self.moving_avg = moving_avg_enhanced(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)

        res = x - moving_mean
        return res, moving_mean


class PDDFNet_layer(L.LightningModule):
    # dim: embed_dim
    def __init__(self, dim, groups, kernel_size=5, drop=0.3, drop_path=0.):
        super().__init__()

        self.drop_out = DropPath(drop) if drop_path > 0. else nn.Identity()

        self.groups = groups
        self.FeatureInfoEncoders = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.groups + 1)])
        self.FeatureEncoders = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.groups + 1)])
        self.ConfidenceLayers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.groups + 1)])

        self.var_gate = nn.Sequential(
            nn.Linear(args.emb_dim, 1),
            nn.Sigmoid()
        )

        # [(batch*ts_d), seg_num, d_model]
        self.decomp = InterPatchSeriesDecomposition(kernel_size=kernel_size, adaptive=False)
        self.gate_controller = nn.Sequential(
            nn.Linear(2 * dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
            nn.Softmax(dim=-1)
        )

        self.ipfeb = IntraPatchFrequencyEnhancement(dim)


    def forward(self, x):

        n = x.shape[2]
        m = x.shape[1]
        x = rearrange(x, 'b m n p -> (b m) n p')

        if args.IPSDB and args.IPFEB:

            seasonal, trend = self.decomp(x)

            gate_input = torch.cat([seasonal, trend], dim=-1)
            gates = self.gate_controller(gate_input)  # [B*M, N, 2]
            fused = gates[..., 0:1] * seasonal + gates[..., 1:2] * trend

            y = self.ipfeb(fused)
            x = x + y


        elif args.IPSDB:
            seasonal, trend = self.decomp(x)  # each [B*M, N, P]
            gate_input = torch.cat([seasonal, trend], dim=-1)
            gates = self.gate_controller(gate_input)  # [B*M, N, 2]
            fused = gates[..., 0:1] * seasonal + gates[..., 1:2] * trend
            x = x + fused

        elif args.IPFEB:
            x = x + self.ipfeb(x)


        x = rearrange(x, '(b m) n p -> (b n) m p', m=m) # [(batch*seg_num), ts_d, d_model]，model relation among variables

        ts_d = x.shape[1]
        group_size = ts_d // self.groups
        remainder = ts_d % self.groups

        FeatureInfo, feature, TCPLogit, Confidence = {}, {}, {}, {}

        for g in range(self.groups):
            start, end = g * group_size, (g + 1) * group_size
            FeatureInfo[g] = torch.sigmoid(self.FeatureInfoEncoders[g](x[:, start:end, :]))
            # FeatureInfo {group}[batch*seg_num, ts_d / group, d_model]
            feature[g] = x[:, start:end, :] * FeatureInfo[g]
            feature[g] = self.FeatureEncoders[g](feature[g])
            feature[g] = F.relu(feature[g])
            feature[g] = self.drop_out(feature[g])
            # Feature [batch*seg_num, ts_d / group, d_model]
            Confidence[g] = torch.sigmoid(self.ConfidenceLayers[g](feature[g]))
            feature[g] = feature[g] * Confidence[g]

        if remainder > 0:
            FeatureInfo[self.groups] = torch.sigmoid(self.FeatureInfoEncoders[-1](x[:, -remainder:, :]))
            feature[self.groups] = x[:, -remainder:, :] * FeatureInfo[self.groups]
            feature[self.groups] = self.FeatureEncoders[-1](feature[self.groups])
            feature[self.groups] = F.relu(feature[self.groups])
            feature[self.groups] = self.drop_out(feature[self.groups])

            Confidence[self.groups] = torch.sigmoid(self.ConfidenceLayers[-1](feature[self.groups]))
            feature[self.groups] = feature[self.groups] * Confidence[self.groups]

        final_feature = torch.cat(
            [feature[g] for g in range(self.groups)] + ([feature[self.groups]] if remainder > 0 else []), dim=1
        )

        # (b seg_num) ts_d d_model
        x = rearrange(final_feature, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', seg_num=n)

        gate = self.var_gate(x.mean(dim=2))  # [b, ts_d, 1]
        weighted_x = x * gate.unsqueeze(2)  # [b, ts_d, seg_num, d_model]

        x = weighted_x
        return x


class PDDFNet(nn.Module):

    def __init__(self):
        super(PDDFNet, self).__init__()

        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)

        self.data_dim = args.data_dim
        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.pddf_blocks = nn.ModuleList([
            PDDFNet_layer(dim=args.emb_dim, groups=args.groups, kernel_size=args.kernel_size, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # Parameters/Embeddings
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

        # self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, num_patches, args.emb_dim))

    def pretrain(self, x_in):
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        m = x_patched.shape[1]
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch]
        xb_mask = self.input_layer(xb_mask)

        xb_mask = rearrange(xb_mask, '(b m) n p -> b m n p', m=m)

        for pddf_blk in self.pddf_blocks:
            xb_mask = pddf_blk(xb_mask)

        xb_mask = rearrange(xb_mask, 'b m n p -> (b m) n p')

        return xb_mask, self.input_layer(x_patched)


    def forward(self, x):
        B, L, M = x.shape # x [512,96,7] [batch, seq_len, num_channels]

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        x = self.input_layer(x) # nn.Linear(self.patch_size, args.emb_dim)
        for pddf_blk in self.pddf_blocks:
            x = pddf_blk(x)

        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = PDDFNet()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        _, _, C = batch_x.shape
        batch_x = batch_x.float().to(device)

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = PDDFNet()
        # self.criterion_MSE = nn.MSELoss()
        self.criterion_MAE = nn.L1Loss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.preds = []
        self.trues = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-6)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience),
            'monitor': 'val_mae',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)
        loss = self.criterion_MAE(outputs, batch_y)

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        mse = self.mse(pred.contiguous(), true.contiguous())
        mae = self.mae(pred.contiguous(), true.contiguous())

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._calculate_loss(batch, mode="test")
        self.preds.append(preds)
        self.trues.append(trues)
        return {'test_loss': loss, 'pred': preds, 'true': trues}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mse = self.mse(preds.contiguous(), trues.contiguous())
        mae = self.mae(preds, trues)
        print(f"{mae, mse}")

def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=args.early_stop_patience,
        min_delta=0.001,
        mode='min',
        verbose=True,
        strict=True
    )

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=args.train_epochs,
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            early_stop_callback,
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    # model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    mse_result = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
    mae_result = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

    return model, mse_result, mae_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data args...
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--log', type=str, default='ETTm1.txt', help='log path') # etth1_96 = search_new


    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # ori 96
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # 近期趋势比长期历史更重要时候少一些，否则可以和pred_len一样
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4') # 用于factory中的自定义数据集

    # optimization
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs') # 50
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=float, default=2)
    parser.add_argument('--early_stop_patience', type=float, default=100)


    # model
    parser.add_argument('--emb_dim', type=int, default=192, help='dimension of model') # ori 64
    parser.add_argument('--groups', type=int, default=6, help='groups of model') # ori 64
    parser.add_argument('--depth', type=int, default=2, help='num of layers') # ori 3
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout value') # 0.5
    parser.add_argument('--patch_size', type=int, default=8, help='size of patches') # ori 64
    parser.add_argument('--mask_ratio', type=float, default=0.4) # 0.4
    parser.add_argument('--lr', type=float, default=0.0003) # 1e-4

    # PDDFNet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False, help='False: without pretraining')
    parser.add_argument('--IPSDB', type=str2bool, default=True)
    parser.add_argument('--IPFEB', type=str2bool, default=True)
    parser.add_argument('--kernel_size', type=int, default=7)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(0))

    # load from checkpoint
    run_description = f"{args.data_path.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
    run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}"
    run_description += f"_IPFEB_{args.IPFEB}_IPSDB_{args.IPSDB}_preTr_{args.load_from_pretrained}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_mae',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_data, train_loader = data_provider(args, flag='train') # train_data.data_x同data_y [8640,7]
    vali_data, val_loader = data_provider(args, flag='val') # vali_data.data_x同data_y [2976,7]
    test_data, test_loader = data_provider(args, flag='test') # test_data.data_x同data_y [2976,7]
    print("Dataset loaded ...")

    args.data_dim = train_data.data_x.shape[-1]

    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model()
    else:
        best_model_path = ''

    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # train
    model, mse_result, mae_result = train_model(best_model_path)
    print("MSE results", mse_result)
    print("MAE  results", mae_result)

    # Save results into an Excel sheet ...
    df = pd.DataFrame({
        'MSE': mse_result,
        'MAE': mae_result
    })
    df.to_excel(os.path.join(CHECKPOINT_PATH, f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    # Append results into a text file ...
    os.makedirs("textOutput", exist_ok=True)
    f = open(f"textOutput/PDDFNet_{os.path.basename(args.data_path)}.txt", 'a')
    f.write(run_description + "  \n")
    f.write('MSE:{}, MAE:{}'.format(mse_result, mae_result))
    f.write('\n')
    f.write('\n')
    f.close()


