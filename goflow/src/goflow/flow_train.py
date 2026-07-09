from typing import Any, Dict, List, Optional, Tuple
import os
import torch
import lightning as L
from lightning import LightningDataModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
import hydra
from omegaconf import DictConfig
from goflow.flow_matching.flow_module import FlowModule
from goflow.gotennet.models.representation.gotennet import GotenNet
from goflow.gotennet.models.components.ops import CosineCutoff, Distance, NodeInit
from goflow.gotennet.utils import RankedLogger, instantiate_callbacks, instantiate_loggers, log_hyperparameters
from lightning.pytorch.plugins.io import TorchCheckpointIO

# Register custom classes for safe checkpoint loading
torch.serialization.add_safe_globals([GotenNet, CosineCutoff, Distance])

log = RankedLogger(__name__, rank_zero_only=True)

class UnsafeCheckpointIO(TorchCheckpointIO):
    def load_checkpoint(self, path, map_location=None, weights_only=True):
        # Ignore 'weights_only' argument passed by Lightning and force False
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def train_flow(cfg: DictConfig):
    #################### PyTorch Specifics ####################
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    if cfg.get("seed"): L.seed_everything(cfg.seed, workers=True)

    #################### Load Modules for Data/Model/Callback/Logging ####################
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: FlowModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, plugins=[UnsafeCheckpointIO()])

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    #################### Run Training or Testing ####################
    if cfg.get("train"):
        log.info("Starting training!")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is not None:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint path provided but not found: {ckpt_path}")
            log.info(f"Loading model weights from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = cfg.get("custom_model_weight_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


if __name__ == '__main__':
    train_flow()
