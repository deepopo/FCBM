import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from utils import instantiate_from_config
from utils.analysis import js_div
from utils import cos_similarity_cubed_single, zero_out_small_weights

class FCBMConceptInterface(pl.LightningModule):
    def __init__(self, model_config, text_features):
        super().__init__()
        self.save_hyperparameters()
        
        self.values = dict() # log_dict
        self.fcbm_backbone = instantiate_from_config(model_config.backbone_config)
        self.register_buffer('text_features', text_features)
        self.text_features = self.text_features.to(self.device)

        self.num_concepts = model_config.backbone_config.params.num_concepts
        self.lr = model_config.concept_lr
        self.proj_batch_size = model_config.proj_batch_size
        self.dataset = model_config.dataset
        self.save_dir = model_config.save_dir
        self.backbone = model_config.backbone
        self.clip_name = model_config.clip_name

    def forward(self, batch, trans=False):
        target_features, clip_features, labels = batch[0], batch[1], batch[2]
        return target_features, clip_features, labels

    def training_step(self, batch, batch_idx):
        self.fcbm_backbone.train()

        target_features, clip_features, _ = self(batch)

        outs_c = self.fcbm_backbone(target_features)
        loss_c = -cos_similarity_cubed_single(clip_features.detach(), outs_c)
        loss_c = torch.mean(loss_c)
        # self.log_util(-loss_c.item(), 'sim_c')
        # self.log_util(loss_c.item(), 'loss_c')
        self.train_loss_c_accum += loss_c.item() * clip_features.size(0)
        self.train_total_samples += clip_features.size(0)
        
        return loss_c

    def on_train_epoch_start(self):
        self.train_loss_c_accum = 0.
        self.train_total_samples = 0

    def on_train_epoch_end(self):
        if self.train_total_samples != 0:
            self.log_util(self.train_loss_c_accum/self.train_total_samples, 'loss_c')
            self.log_util(-self.train_loss_c_accum/self.train_total_samples, 'sim_c')

    def validation_step(self, batch, batch_idx):
        self.fcbm_backbone.eval()

        target_features, clip_features, _ = self(batch)
        outs_c = self.fcbm_backbone(target_features)
        loss_c = -cos_similarity_cubed_single(clip_features.detach(), outs_c)
        loss_c = torch.mean(loss_c)
        self.val_loss_c_accum += loss_c.item() * clip_features.size(0)
        self.val_total_samples += clip_features.size(0)

    def on_validation_epoch_start(self):
        self.val_loss_c_accum = 0.
        self.val_total_samples = 0

    def on_validation_epoch_end(self):
        self.log_util(self.val_loss_c_accum/self.val_total_samples, 'val_loss_c')
        self.log_util(-self.val_loss_c_accum/self.val_total_samples, 'val_sim_c')

    def test_step(self, batch, batch_idx):
        self.fcbm_backbone.eval()

        target_features, clip_features, _ = self(batch)
        outs_c = self.fcbm_backbone(target_features)
        print(outs_c.shape)
        loss_c = -cos_similarity_cubed_single(clip_features.detach(), outs_c)
        loss_c = torch.mean(loss_c)
        self.test_loss_c_accum += loss_c.item() * clip_features.size(0)
        self.test_total_samples += clip_features.size(0)
        self.outs_c_list.append(outs_c.detach().cpu())

    def on_test_epoch_start(self):
        self.test_loss_c_accum = 0.
        self.test_total_samples = 0
        self.outs_c_list = []

    def on_test_epoch_end(self):
        ordered_train_dataloader = torch.utils.data.DataLoader(self.trainer.datamodule.train_dataset, 
                                                               batch_size=self.trainer.test_dataloaders.batch_size,
                                                               num_workers=self.trainer.test_dataloaders.num_workers, 
                                                               persistent_workers=True,
                                                               shuffle=False,  # affirm that the order is correct
                                                               pin_memory=True)
        self.eval()
        self.save_tensors_incrementally(ordered_train_dataloader, 'train')
        self.log_util(-self.test_loss_c_accum/self.test_total_samples, 'test_sim_c')
        if not os.path.exists(f'{self.save_dir}{self.dataset}_test_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}.pt'):
            torch.save(torch.cat(self.outs_c_list, 0), f'{self.save_dir}{self.dataset}_test_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}.pt')
        self.log_util(self.outs_c_list[0].size(1), 'num_concepts')

    def save_tensors_incrementally(self, dataloader, mode):
        """Save tensors incrementally by batch index to avoid memory issues"""
        save_dir = f'{self.save_dir}{self.dataset}_{mode}_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}_batches/'
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata file to track number of batches
        metadata_path = os.path.join(save_dir, 'metadata.txt')
        
        with torch.no_grad():
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                target_features, clip_features, _ = self(batch)
                outs_c = self.fcbm_backbone(target_features)
                
                # Save each batch as a separate file
                batch_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
                torch.save(outs_c.detach().cpu(), batch_path)
                batch_count += 1
                
        # Save metadata
        with open(metadata_path, 'w') as f:
            f.write(str(batch_count))
        
        print(f"Successfully saved {batch_count} batches to {save_dir}")

    def _obtain_c(self, dataloader, mode='train'):
        save_path = f'{self.save_dir}{self.dataset}_{mode}_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}.pt'
        if not os.path.exists(save_path):
            with torch.no_grad(), open(save_path, 'wb') as f:
                for batch in dataloader:
                    batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                    target_features, clip_features, _ = self(batch)
                    outs_c = self.fcbm_backbone(target_features)
                    torch.save(outs_c.detach().cpu(), f)

    def log_util(self, loss, name='loss'):
        self.values[name] = loss
        self.log_dict(self.values, logger=True, prog_bar=True, on_step=False, on_epoch=True, 
                      batch_size=self.proj_batch_size)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.fcbm_backbone.parameters(), lr=self.lr)
        return [optimizer], []

class FCBMHyperInterface(pl.LightningModule):
    def __init__(self, model_config, text_features):
        super().__init__()
        self.save_hyperparameters()
        
        self.values = dict() # log_dict
        self.vlm_hyper = instantiate_from_config(model_config.hyper_config)
        self.register_buffer('text_features', text_features)
        self.text_features = self.text_features.to(self.device)

        self.lr = model_config.hyper_lr
        self.hyper_batch_size = model_config.hyper_batch_size
        self.sparse = model_config.sparse
        self.train_concept_prop = model_config.train_concept_prop
        self.test_zero_shot = model_config.test_zero_shot
        self.act_thred = model_config.act_thred
        self.decay_rate = model_config.decay_rate
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.early_stop_trigger = 0

    def forward(self, batch, trans=False):
        outs_c, labels = batch[0], batch[1]
        return outs_c, labels

    def training_step(self, batch, batch_idx):
        self.vlm_hyper.train()

        outs_c, labels = self(batch)
        outs_y = self.vlm_hyper(outs_c, self.text_features, self.sparse, self.train_concept_prop, self.test_zero_shot)
    
        loss_y = F.cross_entropy(outs_y, labels)

        # self.log_util(loss_y, 'train_loss')
        self.train_loss_y_accum += loss_y.item() * outs_c.size(0)
        self.train_total_samples += outs_c.size(0)
        return loss_y

    def on_train_epoch_start(self):
        self.train_loss_y_accum = 0.
        self.train_total_samples = 0

    def on_train_epoch_end(self):
        if self.train_total_samples != 0:
            self.log_util(self.train_loss_y_accum/self.train_total_samples, 'train_loss')
        # temperature decay
        if self.vlm_hyper.avg_act_concepts > self.act_thred:
            shaped_temperature = (self.vlm_hyper.temperature.data - 1.) / 9.
            shaped_temperature = shaped_temperature * self.decay_rate
            self.vlm_hyper.temperature.data = 1.0 + 9. * shaped_temperature

    def validation_step(self, batch, batch_idx):
        self.vlm_hyper.eval()
        outs_c, labels = self(batch)
        outs_y = self.vlm_hyper(outs_c, self.text_features, self.sparse, self.train_concept_prop, self.test_zero_shot)
        loss_y = F.cross_entropy(outs_y, labels)

        self.val_loss_y_accum += loss_y.item() * outs_c.size(0)
        _, pred = outs_y.topk(1, 1, True, True)
        pred = pred.t()
        labels = labels.view(1, -1).expand_as(pred)
        correct = pred.eq(labels)
        correct = correct[:1].reshape(-1).float().mean(0, keepdim=True) * 100
        self.val_correct_accum += correct.item() * outs_c.size(0)
        self.val_total_samples += outs_c.size(0)

    def on_validation_epoch_start(self):
        self.val_correct_accum = 0.
        self.val_loss_y_accum = 0.
        self.val_total_samples = 0

    def on_validation_epoch_end(self):
        self.log_util(self.val_correct_accum/self.val_total_samples, 'val_acc')
        self.log_util(self.val_loss_y_accum/self.val_total_samples, 'val_loss_y')
        if self.vlm_hyper.avg_act_concepts < self.act_thred:
            self.early_stop_trigger += 1
            self.log_util(-self.val_correct_accum/self.val_total_samples, 'val_loss')
        else:
            if self.vlm_hyper.sparse:
                self.log_util(self.vlm_hyper.avg_act_concepts, 'val_loss')
            else:
                self.log_util(-self.val_correct_accum/self.val_total_samples, 'val_loss')
        if hasattr(self, "early_stop_trigger") and self.early_stop_trigger > 5 and self.vlm_hyper.avg_act_concepts > self.act_thred:
            print(f"Re-existing avg_act_concepts > {self.act_thred}. Early Stop!")
            self.trainer.should_stop = True
        self.log_util(self.vlm_hyper.scale_factor, 'scale_factor')
        self.log_util(self.vlm_hyper.temperature, 'temperature')
        self.log_util(self.vlm_hyper.avg_act_concepts, 'val_avg_act_concepts')

    def test_step(self, batch, batch_idx):
        self.vlm_hyper.eval()

        outs_c, labels = self(batch)
        # s = 0
        # for i in range(15):
        #     s += (labels == i).sum()
        #     print(i, s)
        outs_y = self.vlm_hyper(outs_c, self.text_features, self.sparse, self.train_concept_prop, self.test_zero_shot)
        _, pred = outs_y.topk(1, 1, True, True)
        pred = pred.t()
        labels = labels.view(1, -1).expand_as(pred)
        correct = pred.eq(labels)
        print(pred[0, 328+11])
        print(labels[0, 328+11])
        print(pred[0, 328: 328+20])
        print(labels[0, 328: 328+20])
        # import sys
        # sys.exit()
        # print(labels[0, 11])
        # a = correct[0][:100].detach().cpu().numpy().tolist()
        # print(correct[0][:100].detach().cpu().numpy().tolist())
        # import numpy as np
        # a = np.array(a)
        # print(np.where(a))
        # import sys
        # sys.exit()
        correct = correct[:1].reshape(-1).float().mean(0, keepdim=True) * 100
        self.test_correct_accum += correct.item() * outs_c.size(0)
        self.test_total_samples += outs_c.size(0)

    def on_test_epoch_start(self):
        self.test_correct_accum = 0.
        self.test_total_samples = 0

    def on_test_epoch_end(self):
        self.log_util(self.test_correct_accum/self.test_total_samples, 'test_acc')
        self.log_util(self.text_features.size(0), 'total_concepts')
        self.log_util(self.vlm_hyper.scale_factor, 'scale_factor')
        self.log_util(self.vlm_hyper.temperature, 'temperature')
        self.log_util(self.vlm_hyper.avg_act_concepts, 'test_avg_act_concepts')

    def log_util(self, loss, name='loss'):
        self.values[name] = loss
        self.log_dict(self.values, logger=True, prog_bar=True, on_step=False, on_epoch=True, 
                      batch_size=self.hyper_batch_size)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vlm_hyper.parameters(), lr=self.lr)
        return [optimizer], []