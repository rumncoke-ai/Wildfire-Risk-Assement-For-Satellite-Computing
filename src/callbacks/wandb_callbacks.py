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
from sklearn.metrics import f1_score, precision_score, recall_score
import gc
from thop import profile


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


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
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
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


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


# class LogModelFLOPs(Callback):
#     """
#     Calculates the FLOPs (Floating Point Operations) and Parameters of the model
#     at the start of the run and logs them to WandB summary.
#     """
#     def __init__(self):
#         self.calculated = False

#     @rank_zero_only
#     def on_train_start(self, trainer, pl_module):
#         if self.calculated:
#             return

#         try:
#             import thop
#         except ImportError:
#             print("❌ 'thop' library not found. Install it via: pip install thop")
#             return

#         # 1. Grab a single batch from the training dataloader
#         try:
#             # We fetch one batch to get the correct input shapes (batch_size, time, channels, etc.)
#             dl = trainer.train_dataloader
#             batch = next(iter(dl))
            
#             # Ensure batch is on the correct device
#             batch = [t.to(pl_module.device) if torch.is_tensor(t) else t for t in batch]
#         except Exception as e:
#             print(f"⚠️ Could not load a batch for FLOPs calculation: {e}")
#             return

#         # 2. Register a hook to capture the processed input
#         # We do this because your LightningModule.step() does complex processing 
#         # (combining dynamic+static) before passing it to self.model.
#         # We want to measure self.model using the FINAL combined tensor.
#         captured_input = []

#         def hook_fn(module, inputs, output):
#             # Inputs is a tuple; we take the first element (the tensor x)
#             captured_input.append(inputs[0])

#         # Attach hook to the inner model (SimpleCNN or SimpleLSTM)
#         handle = pl_module.model.register_forward_hook(hook_fn)

#         # 3. Run a "Dry" Forward Pass
#         pl_module.eval() # Set to eval to avoid updating batchnorm stats
#         with torch.no_grad():
#             try:
#                 # Trigger the forward pass logic defined in your training_step
#                 pl_module.training_step(batch, batch_idx=0)
#             except Exception as e:
#                 # This might fail if the step relies on specific optimizer states, 
#                 # but usually it's fine for a single pass.
#                 print(f"⚠️ Dry run for FLOPs failed: {e}")
        
#         # Cleanup: Remove hook and reset mode
#         handle.remove()
#         pl_module.train()

#         # 4. Calculate FLOPs using the captured input
#         if captured_input:
#             try:
#                 input_t = captured_input[0]
                
#                 # thop.profile returns (MACs, Params). FLOPs ~= 2 * MACs
#                 macs, params = thop.profile(pl_module.model, inputs=(input_t, ), verbose=False)
                
#                 gflops = macs / 1e9
#                 mparams = params / 1e6

#                 print(f"\n📊 Model Efficiency: {gflops:.4f} GFLOPs | {mparams:.4f} MParams\n")

#                 # 5. Log to WandB Summary (Best for single-value comparison)
#                 logger = get_wandb_logger(trainer)
#                 # Using .summary ensures it appears at the top level of the run dashboard
#                 logger.experiment.summary["GFLOPs"] = gflops
#                 logger.experiment.summary["Params_M"] = mparams
                
#                 self.calculated = True
                
#             except Exception as e:
#                 print(f"⚠️ Error calculating FLOPs with thop: {e}")
#         else:
#             print("⚠️ Could not capture model input. FLOPs not calculated.")

class LogModelFLOPs(Callback):
    def __init__(self):
        self.calculated = False

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        if self.calculated:
            return

        try:
            import thop
        except ImportError:
            print("❌ Install thop: pip install thop")
            return

        # --- Get one batch safely ---
        try:
            dl = trainer.train_dataloader()
            batch = next(iter(dl))
        except Exception as e:
            print(f"⚠️ Could not load batch: {e}")
            return

        def move(obj):
            if torch.is_tensor(obj):
                return obj.to(pl_module.device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(move(o) for o in obj)
            if isinstance(obj, dict):
                return {k: move(v) for k, v in obj.items()}
            return obj

        batch = move(batch)

        # --- Capture final model input ---
        captured_input = []

        def hook_fn(module, inputs, output):
            if inputs and torch.is_tensor(inputs[0]):
                captured_input.append(inputs[0])

        handle = pl_module.model.register_forward_hook(hook_fn)

        pl_module.eval()
        with torch.no_grad():
            try:
                pl_module.training_step(batch, batch_idx=0)
            except Exception as e:
                print(f"⚠️ Dry run failed: {e}")

        handle.remove()
        pl_module.train()

        if not captured_input:
            print("⚠️ No input captured for FLOPs")
            return

        x = captured_input[0][:1]  # batch size = 1

        macs, params = thop.profile(pl_module.model, inputs=(x,), verbose=False)
        gflops = 2 * macs / 1e9
        mparams = params / 1e6

        logger = get_wandb_logger(trainer)
        logger.experiment.summary["GFLOPs"] = gflops
        logger.experiment.summary["Params_M"] = mparams

        print(f"\n📊 GFLOPs: {gflops:.4f} | Params: {mparams:.4f}M\n")

        self.calculated = True
