import glob
import os
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score
import gc
from thop import profile
import copy
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis



def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.log_artifact(ckpts)


# class LogConfusionMatrix(Callback):
#     """Generate confusion matrix every epoch and send it to wandb.
#     Expects validation step to return predictions and targets.
#     """

#     def __init__(self):
#         self.preds = []
#         self.targets = []
#         self.ready = True

#     def on_sanity_check_start(self, trainer, pl_module) -> None:
#         self.ready = False

#     def on_sanity_check_end(self, trainer, pl_module):
#         """Start executing this callback only after all validation sanity checks end."""
#         self.ready = True

#     def on_validation_batch_end(
#             self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         """Gather data from single batch."""
#         if self.ready:
#             self.preds.append(outputs["preds"])
#             self.targets.append(outputs["targets"])

#     def on_validation_epoch_end(self, trainer, pl_module):
#         """Generate confusion matrix."""
#         if self.ready:
#             logger = get_wandb_logger(trainer)
#             experiment = logger.experiment

#             preds = torch.cat(self.preds).cpu().numpy()
#             targets = torch.cat(self.targets).cpu().numpy()

#             confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

#             # set figure size
#             plt.figure(figsize=(14, 8))

#             # set labels size
#             sn.set(font_scale=1.4)

#             # set font size
#             sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")
#             plt.xlabel('Predicted')
#             plt.ylabel('True')

#             # names should be uniqe or else charts from different experiments in wandb will overlap
#             experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

#             # according to wandb docs this should also work but it crashes
#             # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

#             # reset plot
#             plt.clf()

#             self.preds.clear()
#             self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()

class LogAllMetrics(Callback):
    """
    Comprehensive metric logging callback that tracks ALL metrics from the paper:
    - TP, FP, TN, FN
    - Precision, Recall, F1-score
    - AUROC, OA (Overall Accuracy)
    - Per-class metrics
    """
    
    def __init__(self):
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []  # For AUROC
        self.train_losses = []
        self.val_losses = []
        self.ready = True
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track training loss."""
        if self.ready and outputs and "loss" in outputs:
            self.train_losses.append(outputs["loss"].item())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Gather validation data from single batch."""
        if self.ready:
            # Store predictions (class labels)
            self.val_preds.append(outputs["preds"])
            # Store targets
            self.val_targets.append(outputs["targets"])
            # Store probabilities for AUROC (if available)
            if "probs" in outputs:
                self.val_probs.append(outputs["probs"])
            # Store validation loss if available
            if "val_loss" in outputs:
                self.val_losses.append(outputs["val_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Calculate and log ALL metrics at the end of validation epoch."""
        if self.ready and len(self.val_preds) > 0:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            # Concatenate all predictions and targets
            preds = torch.cat(self.val_preds).cpu().numpy()
            targets = torch.cat(self.val_targets).cpu().numpy()
            
            # Calculate all metrics
            # 1. Confusion Matrix Components
            cm = confusion_matrix(y_true=targets, y_pred=preds)
            tn, fp, fn, tp = cm.ravel()
            
            # 2. Core Metrics
            precision = precision_score(targets, preds, average='binary', zero_division=0)
            recall = recall_score(targets, preds, average='binary', zero_division=0)
            f1 = f1_score(targets, preds, average='binary', zero_division=0)
            accuracy = accuracy_score(targets, preds)
            
            # 3. AUROC (if probabilities are available)
            auroc = 0.0
            if len(self.val_probs) > 0:
                probs = torch.cat(self.val_probs).cpu().numpy()
                # Assuming binary classification with probability of positive class
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]  # Take probability of positive class
                auroc = roc_auc_score(targets, probs)
            
            # 4. Calculate average losses
            avg_train_loss = np.mean(self.train_losses) if self.train_losses else 0
            avg_val_loss = np.mean(self.val_losses) if self.val_losses else 0
            
            # Log ALL metrics to wandb (matching paper's Table I & II format)
            metrics_dict = {
                # Confusion matrix components
                "val/tp": int(tp),
                "val/fp": int(fp),
                "val/tn": int(tn),
                "val/fn": int(fn),
                
                # Core classification metrics
                "val/precision": precision * 100,  # Convert to percentage like paper
                "val/recall": recall * 100,
                "val/f1": f1 * 100,
                "val/auroc": auroc * 100,
                "val/accuracy": accuracy * 100,  # This is OA (Overall Accuracy)
                
                # Losses
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                
                # Additional useful metrics
                "val/true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "val/false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "val/specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            }
            
            experiment.log(metrics_dict, commit=False)
            
            # Log confusion matrix as image
            self._log_confusion_matrix(experiment, cm, targets, preds)
            
            # Clear all stored values
            self.val_preds.clear()
            self.val_targets.clear()
            self.val_probs.clear()
            self.train_losses.clear()
            self.val_losses.clear()
    
    def _log_confusion_matrix(self, experiment, cm, targets, preds):
        """Log confusion matrix as an image."""
        plt.figure(figsize=(10, 8))
        sn.set(font_scale=1.2)
        
        # Create annotated confusion matrix with percentages
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot with both counts and percentages
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_percentage[i, j]
                annot[i, j] = f'{c}\n({p:.1f}%)'
        
        sn.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                  xticklabels=['Negative', 'Positive'],
                  yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})')
        
        experiment.log({"confusion_matrix": wandb.Image(plt)}, commit=False)
        plt.close()


class LogModelEfficiency(Callback):

    def __init__(self):
        self.logged = False

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):

        if self.logged:
            return

        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        # Work on CPU copy to avoid CUDA issues
        eval_model = copy.deepcopy(pl_module).cpu()
        eval_model.eval()

        total_params = sum(p.numel() for p in eval_model.parameters())
        trainable_params = sum(
            p.numel() for p in eval_model.parameters() if p.requires_grad
        )

        mmacs = 0
        samples_per_ms = 0

        try:
            if not hasattr(pl_module, "example_input_array"):
                raise ValueError("Model must define example_input_array for profiling.")

            example_input = pl_module.example_input_array.cpu()

            with torch.no_grad():

                # -------- FLOPs / MACs --------
                flops = FlopCountAnalysis(eval_model, example_input)
                total_flops = flops.total()

                # Convert FLOPs → MACs (1 MAC = 2 FLOPs)
                macs = total_flops / 2
                mmacs = macs / 1e6

                # -------- Throughput --------
                for _ in range(10):  # warmup
                    _ = eval_model(example_input)

                n_samples = 100
                start_time = time.time()

                for _ in range(n_samples):
                    _ = eval_model(example_input)

                total_time_ms = (time.time() - start_time) * 1000
                samples_per_ms = (
                    n_samples / total_time_ms if total_time_ms > 0 else 0
                )

        except Exception as e:
            print(f"Efficiency logging failed: {e}")

        efficiency_metrics = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/mmacs": mmacs,
            "model/samples_per_ms": samples_per_ms,
            "model/inference_time_ms_per_sample":
                (1 / samples_per_ms) if samples_per_ms > 0 else 0,
        }

        experiment.config.update(efficiency_metrics)
        experiment.log(efficiency_metrics, commit=False)

        del eval_model
        self.logged = True


class LogVariableImportance(Callback):
    """
    Log which variables are being used (for ablation studies on limited variables).
    This helps track your "limited variables" experiment.
    """
    
    def __init__(self, dynamic_features: List[str], static_features: List[str]):
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Log variable configuration."""
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        
        # Log variable counts and names
        variable_info = {
            "data/num_dynamic_variables": len(self.dynamic_features),
            "data/num_static_variables": len(self.static_features),
            "data/total_variables": len(self.dynamic_features) + len(self.static_features),
            "data/dynamic_variables": ", ".join(self.dynamic_features),
            "data/static_variables": ", ".join(self.static_features),
        }
        
        experiment.config.update(variable_info)
        experiment.log(variable_info, commit=False)


# Setup function to initialize wandb with proper config
def setup_wandb(
    project_name: str,
    run_name: str,
    model_name: str,
    dynamic_features: List[str],
    static_features: List[str],
    neg_pos_ratio: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    config_dict: dict = None
):
    """
    Initialize wandb with all configuration parameters.
    Call this before creating your trainer.
    """
    
    # Base configuration
    config = {
        "model_name": model_name,
        "dynamic_features": dynamic_features,
        "static_features": static_features,
        "num_dynamic": len(dynamic_features),
        "num_static": len(static_features),
        "total_variables": len(dynamic_features) + len(static_features),
        "neg_pos_ratio": neg_pos_ratio,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    
    # Add any additional config
    if config_dict:
        config.update(config_dict)
    
    # Initialize wandb
    wandb.init(project=project_name, name=run_name, config=config)
    
    return WandbLogger(project=project_name, name=run_name, log_model=True)
