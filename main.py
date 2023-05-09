from typing import Any
import argparse
from omegaconf import OmegaConf
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from modules.ProjectionLayer import ProjectionLayer
from modules.EncoderLayer import EncoderLayer
from PreProcess import DataModules


class ModulePLStyle(LightningModule):
    def __init__(self,  num_classes: int, lr: float, weight_decay: float,
                 embedding_dim: int = 420, max_len: int = 64, hidden_dim: int = 768,
                 num_heads: int = 8, n_layers: int = 2, dropout_rate: float = 0.4, *args, **kwargs):
        super(ModulePLStyle, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            ProjectionLayer(max_len, embedding_dim, hidden_dim),
            EncoderLayer(hidden_dim, hidden_dim, num_heads, n_layers, dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=num_classes),
            # batch_size * 420 to batch_size * 2
            nn.Softmax(dim=-1)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.train_metrices = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metrices = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_metrices = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        # opt = self.optimizers()
        logits = self.classifier(self.model(batch))
        loss = self.loss(logits, batch["labels"].long().view(-1))
        logits_indices = torch.argmax(logits, dim=-1)
        train_metrics_batch = self.train_metrices(logits_indices, batch["labels"].view(-1))
        self.log("train-loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train-metrics", train_metrics_batch, prog_bar=True, on_step=True, on_epoch=False)
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()
        return loss

    def on_train_epoch_end(self):
        total_train_metrics = self.train_metrices.compute()
        self.log("train-metrics-epoch", total_train_metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.train_metrices.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:

        # aaa= self.model(batch)
        # print(aaa[0])
        # logits = self.classifier(aaa)
        logits = self.classifier(self.model(batch))
        loss = self.loss(logits, batch["labels"].long().view(-1))
        logits_indices = torch.argmax(logits, dim=-1)
        val_metrics_batch = self.val_metrices(logits_indices, batch["labels"].view(-1))
        self.log("val-loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("batch-val-metrics", val_metrics_batch, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        total_val_metrics = self.val_metrices.compute()
        self.log("val-metrics", total_val_metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrices.reset()

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self.classifier(self.model(batch))
        logits_indices = torch.argmax(logits, dim=-1)
        test_metrics_batch = self.test_metrices(logits_indices, batch["labels"].long().view(-1))
        self.log("test-metrics", test_metrics_batch, prog_bar=True, on_step=True, on_epoch=True)
        return logits_indices

    def predict_step(self, batch_idx, feature, dataloader_idx=0):
        logits = self.classifier(self.model(feature))
        logits_indices = torch.argmax(logits, dim=-1)
        return logits_indices

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-t', '--train', type=str, default="train")
    args.add_argument('-d', "--dataset", type=str, default="cola")
    args.add_argument('-s', "--text", type=str, default="hello word")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # modelConfigs = OmegaConf.load("configs/model/textcnn.yml")
    modelConfigs = OmegaConf.load("configs/model/proformer.yml")
    dataConfigs = OmegaConf.load(f"configs/dataset/{args.dataset}.yml")
    model = ModulePLStyle(num_classes=dataConfigs.num_classes, **modelConfigs).to(device=torch.device("cuda"))
    data = DataModules.DataModule(**dataConfigs)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt), strict=False)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val-metrics',
                filename='mixer-best-{epoch:03d}-{val-metrics:.3f}',
                save_top_k=1,
                mode='max',
                save_last=True
            ),
        # early stopping
        # pl.callbacks.early_stopping.EarlyStopping(
        #     monitor="val_acc",
        #     min_delta=0.001,
        #     mode='max'
        # )
        ],
        enable_checkpointing=True,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger("logs/", args.dataset),
        max_epochs=60,
        check_val_every_n_epoch=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.1
    )
    if args.train == 'train':
        trainer.fit(model, data)
    if args.train == 'test':
        trainer.test(model, data, ckpt_path=args.ckpt)
    if args.train == "predict":
        trainer.predict(model, data, ckpt_path=args.ckpt)
