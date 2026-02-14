from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.fire_modules import SimpleLSTM, SimpleConvLSTM, SimpleCNN, Simple1DCNN, MobileNetV2CNN 


def combine_dynamic_static_inputs(dynamic, static, clc, access_mode):
    assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
    if access_mode == 'spatial':
        dynamic = dynamic.float()
        static = static.float()
        input_list = [dynamic, static]
        inputs = torch.cat(input_list, dim=1)
    elif access_mode == 'temporal':
        bsize, timesteps, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)
        input_list = [dynamic, static]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    else:
        bsize, timesteps, _, _, _ = dynamic.shape
        static = static.unsqueeze(dim=1)
        repeat_list = [1 for _ in range(static.dim())]
        repeat_list[1] = timesteps
        static = static.repeat(repeat_list)
        input_list = [dynamic, static]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    return inputs

class ConvLSTM_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 1,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.5,
            access_mode='spatiotemporal',
            clc='vec'
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SimpleConvLSTM(hparams=self.hparams)

        B = 256        
        T = 10       
        C = len(self.hparams.dynamic_features) + len(self.hparams.static_features) + 10
        H = 25       
        W = 25      

        self.example_input_array = torch.zeros(B, T, C, H, W)

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()
        if not self.hparams['clc']:
            clc = None
        inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatiotemporal')
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'

        # log train metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}




class LSTM_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 3,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            attention: bool = False,
            dropout: float = 0.5,
            access_mode='temporal',
            clc='vec'
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.attention = attention
        # if self.attention:
        #     self.model = SimpleLSTMAttention(hparams=self.hparams)
        # else:
        #     self.model = SimpleLSTM(hparams=self.hparams)
        self.model = SimpleLSTM(hparams=self.hparams)

        B = 256        
        T = 10       
        C = len(self.hparams.dynamic_features) + len(self.hparams.static_features) + 10     

        self.example_input_array = torch.zeros(B, T, C)

        self.weight_decay = weight_decay
        # loss function
        
        # self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        class_weights = torch.tensor([1. - positive_weight, positive_weight])
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.NLLLoss(weight=self.class_weights)
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary") # Removed pos_label
        self.train_auprc = AveragePrecision(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary") # Removed pos_label
        self.val_auprc = AveragePrecision(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary") # Removed pos_label
        self.test_auprc = AveragePrecision(task="binary")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()

        if not self.hparams['clc']:
            clc = None

        device = self.device

        dynamic = dynamic.to(device)
        static = static.to(device)
        y = y.to(device)

        if clc is not None:
            clc = clc.to(device)

        inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'temporal')

        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'
        # log train metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'
        # log train metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def on_test_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        """
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CNN_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.5,
            # access_mode='spatiotemporal',
            access_mode='spatial',
            clc=None
    ):
        super().__init__()
        # Saves arguments to self.hparams
        self.save_hyperparameters()

        self.model = SimpleCNN(hparams=self.hparams)

        B = 256        
        C = len(self.hparams.dynamic_features) + len(self.hparams.static_features)
        H = 25       
        W = 25      

        self.example_input_array = torch.zeros(B, C, H, W)

        # Loss function
        # self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        class_weights = torch.tensor([1. - positive_weight, positive_weight])
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.NLLLoss(weight=self.class_weights)

        # Metrics for Train, Validation, and Test
        self.train_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()
        
        if not self.hparams['clc']:
            clc = None

        dynamic = dynamic.to(device)
        static = static.to(device)
        y = y.to(device)

        if clc is not None:
            clc = clc.to(device)

        # inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatiotemporal')
        inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatial')
        
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]

        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # Log metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.hparams.lr_scheduler_step,
            gamma=self.hparams.lr_scheduler_gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
   

class CNN1D_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.5,
            access_mode='temporal',
            clc=None
    ):
        super().__init__()
        # Saves arguments to self.hparams
        self.save_hyperparameters()

        self.model = Simple1DCNN(hparams=self.hparams)

        B = 256        
        C = len(self.hparams.dynamic_features) + len(self.hparams.static_features) + 10
        T = 10           

        self.example_input_array = torch.zeros(B, T, C)

        # Loss function
        # self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        class_weights = torch.tensor([1. - positive_weight, positive_weight])
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.NLLLoss(weight=self.class_weights)

        # Metrics for Train, Validation, and Test
        self.train_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()
        
        if not self.hparams['clc']:
            clc = None

        dynamic = dynamic.to(device)
        static = static.to(device)
        y = y.to(device)

        if clc is not None:
            clc = clc.to(device)

        inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'temporal')
        
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]

        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # Log metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.hparams.lr_scheduler_step,
            gamma=self.hparams.lr_scheduler_gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class MobileNetV2_fire_model(LightningModule):
    """
    MobileNetV2-based model for fire prediction with spatial data.
    """
    def __init__(
            self,
            dynamic_features=None,
            static_features=None,
            hidden_size: int = 32,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.2,
            access_mode='spatial',  # MobileNetV2 works with spatial data
            clc=None,
            freeze_backbone: bool = False  # Option to freeze pretrained layers
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.model = MobileNetV2CNN(hparams=self.hparams)
        
        # Example input for visualization
        B = 256        
        C = len(self.hparams.dynamic_features) + len(self.hparams.static_features)
        H = 25       
        W = 25      
        
        self.example_input_array = torch.zeros(B, C, H, W)
        
        # Freeze backbone if requested (transfer learning)
        if freeze_backbone:
            for param in self.model.mobilenet.features.parameters():
                param.requires_grad = False
        
        # Loss function
        # self.criterion = torch.nn.NLLLoss(
        #     weight=torch.tensor([1. - positive_weight, positive_weight])
        # )
        class_weights = torch.tensor([1. - positive_weight, positive_weight])
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.NLLLoss(weight=self.class_weights)
        
        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")
        
        self.val_accuracy = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        
        self.test_accuracy = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")
    
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()
        
        if not self.hparams['clc']:
            clc = None

        dynamic = dynamic.to(device)
        static = static.to(device)
        y = y.to(device)

        if clc is not None:
            clc = clc.to(device)

        # MobileNetV2 expects spatial inputs
        inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatial')
        
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        
        return loss, preds, preds_proba, y
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        
        # Optional: Log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], 
                 on_step=True, on_epoch=False, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        
        # Log metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def configure_optimizers(self):
        # Use different learning rates for backbone and classifier (if not frozen)
        if self.hparams.get('freeze_backbone', False):
            # Only train the classifier
            optimizer = torch.optim.Adam(
                self.model.mobilenet.classifier.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            # Different learning rates for backbone and classifier
            optimizer = torch.optim.Adam([
                {'params': self.model.mobilenet.features.parameters(), 'lr': self.hparams.lr * 0.1},
                {'params': self.model.mobilenet.classifier.parameters(), 'lr': self.hparams.lr}
            ], weight_decay=self.hparams.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_scheduler_step,
            gamma=self.hparams.lr_scheduler_gamma
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

class TinyTemporalTransformer_fire_model(LightningModule):
    """
    Drop‑in LightningModule consistent with your LSTM/CNN/ConvLSTM structure.
    """
    def __init__(
    self,
    dynamic_features=None,
    static_features=None,
    hidden_size: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 1e-3,
    positive_weight: float = 0.5,
    lr_scheduler_step: int = 10,
    lr_scheduler_gamma: float = 0.1,
    weight_decay: float = 5e-4,
    dropout: float = 0.1,
    access_mode: str = "temporal",
    clc: str | None = "vec",
    ):
        super().__init__()
        self.save_hyperparameters()


        self.model = TinyTemporalTransformer(self.hparams)



        # Example input for Lightning graph tracing
        B, T = 256, 10
        C = len(dynamic_features) + len(static_features) + (10 if clc == "vec" else 0)
        self.example_input_array = torch.zeros(B, T, C)


        # Loss
        # self.criterion = nn.NLLLoss(
        # weight=torch.tensor([1.0 - positive_weight, positive_weight])
        # )
        class_weights = torch.tensor([1. - positive_weight, positive_weight])
        self.register_buffer("class_weights", class_weights)
        self.criterion = torch.nn.NLLLoss(weight=self.class_weights)


        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")


        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")


        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")


    # ------------------------------------------------------------------
    # Forward + shared step
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)


    def step(self, batch: Any):
        dynamic, static, clc, y = batch
        y = y.long()


        if not self.hparams.clc:
            clc = None

        dynamic = dynamic.to(device)
        static = static.to(device)
        y = y.to(device)

        if clc is not None:
            clc = clc.to(device)

        inputs = combine_dynamic_static_inputs(dynamic, static, clc, "temporal")


        logits = self.forward(inputs)
        loss = self.criterion(logits, y)


        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]


        return loss, preds, preds_proba, y


    # ------------------------------------------------------------------
    # Training / Validation / Test
    # ------------------------------------------------------------------


    def training_step(self, batch: Any, _):
        loss, preds, p, y = self.step(batch)


        self.train_accuracy.update(preds, y)
        self.train_auc.update(p, y)
        self.train_auprc.update(p, y)


        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_accuracy, on_epoch=True)
        self.log("train/auc", self.train_auc, on_epoch=True)
        self.log("train/auprc", self.train_auprc, on_epoch=True)
        return loss


    def validation_step(self, batch: Any, _):
        loss, preds, p, y = self.step(batch)


        self.val_accuracy.update(preds, y)
        self.val_auc.update(p, y)
        self.val_auprc.update(p, y)


        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", self.val_accuracy, on_epoch=True)
        self.log("val/auc", self.val_auc, on_epoch=True)
        self.log("val/auprc", self.val_auprc, on_epoch=True)


    def test_step(self, batch: Any, _):
        loss, preds, p, y = self.step(batch)


        self.test_accuracy.update(preds, y)
        self.test_auc.update(p, y)
        self.test_auprc.update(p, y)


        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_accuracy, on_epoch=True)
        self.log("test/auc", self.test_auc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)
    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.Adam(
        self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


        sch = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=self.hparams.lr_scheduler_step,
        gamma=self.hparams.lr_scheduler_gamma,
        )

        return {"optimizer": opt, "lr_scheduler": sch}