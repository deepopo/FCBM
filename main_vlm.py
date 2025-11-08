# -*- coding: utf-8 -*-
import os
import glob
import torch
import argparse
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler

from omegaconf import OmegaConf

from data import VLMDInterface, VLMDHyperInterface
from models import FCBMConceptInterface, FCBMHyperInterface
from utils import load_callbacks, load_hyper_callbacks

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default="cifar10", help='config file')
    parser.add_argument('-seed', type=int, help='seed.')
    return parser.parse_args()

def train(args):
    config = OmegaConf.load(f"configs/{args.d}.yaml")
    seed = args.seed if args.seed else config.base.seed
    pl.seed_everything(seed)
    
    # copy parameters
    concept_mode = config.base.concept_mode
    hyper_mode = config.base.hyper_mode
    dataset = config.base.dataset
    log_dir = config.base.log_dir
    model_name = config.base.model_name
    clip_name = config.base.clip_name
    backbone = config.base.backbone
    max_epochs = config.base.max_epochs
    patience = config.base.patience
    check_val_every_n_epoch = config.base.check_val_every_n_epoch
    proj_batch_size = config.base.proj_batch_size
    hyper_batch_size = config.base.hyper_batch_size
    ckpt_path = None if config.base.ckpt_path == 'None' else config.base.ckpt_path

    config.data.dataset = dataset
    config.data.clip_name = clip_name
    config.data.backbone = backbone
    config.data.batch_size = proj_batch_size
    data_module = VLMDInterface(config.data)

    config.model.backbone_config.params.num_concepts = data_module.num_concepts
    config.model.hyper_config.params.num_class = data_module.num_class
    config.model.proj_batch_size = proj_batch_size
    config.model.hyper_batch_size = hyper_batch_size
    config.model.dataset = dataset
    config.model.save_dir = config.data.save_dir
    config.model.backbone = backbone
    config.model.clip_name = clip_name
    text_features = data_module.text_features
    model = FCBMConceptInterface(config.model, text_features)

    # define logger
    log_name = f'{model_name}_{dataset}_{clip_name.replace("/", "-")}_{backbone.replace("/", "-")}_seed{0}_concept.log'
    root_log_dir = os.path.join(log_dir, log_name)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name)

    # Initialize trainer
    trainer = Trainer(
        accelerator="gpu", 
        max_epochs=max_epochs,
        callbacks=load_callbacks(monitor='val_loss_c', patience=1, mode='min'), 
        logger=logger, 
        check_val_every_n_epoch=check_val_every_n_epoch, 
    )

    # choose to load checkpoints
    if ckpt_path is not None and concept_mode in ['train', 'test', 'both']:
        checkpoint = torch.load(ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    # choose train/test
    if concept_mode in ['train', 'both']:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    if concept_mode in ['test', 'both']:
        trainer = Trainer(accelerator="gpu", callbacks=[], logger=False, enable_checkpointing=False) # don't save checkpoints
        if ckpt_path is None:
            version_dirs = sorted(glob.glob(os.path.join(root_log_dir, "version_*")), key=os.path.getmtime)
            latest_version_dir = version_dirs[-1] if version_dirs else None
            checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
            best_checkpoint = glob.glob(f"{checkpoint_dir}/best-epoch=*.ckpt")
            if best_checkpoint:
                checkpoint_path = best_checkpoint[0]
                print(f"checkpoint path: {checkpoint_path}")
                model = FCBMConceptInterface.load_from_checkpoint(checkpoint_path)
                trainer.test(model, data_module)
            else:
                print("No checkpoint found in the latest version.")
        else:
           trainer.test(model, data_module, ckpt_path=ckpt_path)

    del data_module, model
    config.data.batch_size = hyper_batch_size
    data_module = VLMDHyperInterface(config.data)
    hyper_model = FCBMHyperInterface(config.model, text_features)

    # define logger
    log_name = f'{model_name}_{dataset}_{clip_name.replace("/", "-")}_{backbone.replace("/", "-")}_seed{seed}_label.log'
    root_log_dir = os.path.join(log_dir, log_name)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name)

    profiler = AdvancedProfiler(dirpath=".", filename="advanced_logs")
    # Initialize trainer
    trainer = Trainer(
        accelerator="gpu", 
        max_epochs=max_epochs,
        callbacks=load_callbacks(monitor='val_loss', patience=patience, mode='min'), 
        # callbacks=load_hyper_callbacks(monitor='val_loss', patience=patience, mode='min', act_thred=act_thred), 
        logger=logger, 
        check_val_every_n_epoch=check_val_every_n_epoch, 
        # profiler=profiler
    )

    # choose train/test
    if hyper_mode in ['train', 'both']:
        trainer.fit(hyper_model, data_module, ckpt_path=ckpt_path)
    if hyper_mode in ['test', 'both']:
        trainer = Trainer(accelerator="gpu", callbacks=[], logger=False, enable_checkpointing=False) # don't save checkpoints
        if ckpt_path is None:
            version_dirs = sorted(glob.glob(os.path.join(root_log_dir, "version_*")), key=os.path.getmtime)
            latest_version_dir = version_dirs[-1] if version_dirs else None
            checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
            best_checkpoint = glob.glob(f"{checkpoint_dir}/best-epoch=*.ckpt")
            if best_checkpoint:
                checkpoint_path = best_checkpoint[0]
                print(f"checkpoint path: {checkpoint_path}")
                hyper_model = FCBMHyperInterface.load_from_checkpoint(checkpoint_path)
                hyper_model.train_concept_prop = config.model.train_concept_prop
                hyper_model.test_zero_shot = config.model.test_zero_shot
                trainer.test(hyper_model, data_module)
            else:
                print("No checkpoint found in the latest version.")
        else:
           trainer.test(hyper_model, data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    args = get_args()
    train(args)