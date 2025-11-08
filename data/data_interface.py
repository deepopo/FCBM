import os
import torch
import pytorch_lightning as pl
from torch.utils.data import  DataLoader, TensorDataset

from utils import save_activations, get_save_names, get_save_names
from utils import LABEL_FILES, get_targets_only
from utils import MultiEpochsDataLoader

def load_tensors_incrementally(save_dir):
    """Load tensors from individual batch files and concatenate them"""
    metadata_path = os.path.join(save_dir, 'metadata.txt')
    
    if not os.path.exists(save_dir) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Batch directory or metadata not found at {save_dir}")
    
    # Read metadata
    with open(metadata_path, 'r') as f:
        batch_count = int(f.read().strip())
    
    # Load tensors in chunks to avoid excessive memory usage
    all_tensors = []
    for batch_idx in range(batch_count):
        batch_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
        try:
            tensor = torch.load(batch_path)
            all_tensors.append(tensor)
            
            # Optional: Remove batch file after loading to free disk space
            # os.remove(batch_path)
            
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
    
    # Concatenate all tensors
    if all_tensors:
        return torch.cat(all_tensors, 0)
    else:
        raise RuntimeError("No valid tensor batches were loaded")

class VLMDInterface(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.dataset = data_config.dataset
        self.clip_name = data_config.clip_name
        self.backbone = data_config.backbone
        self.feature_layer = data_config.feature_layer
        self.activation_batch_size = data_config.activation_batch_size
        self.batch_size = data_config.batch_size
        self.save_dir = data_config.save_dir
        self.num_workers = data_config.num_workers
        self.clip_cutoff = data_config.clip_cutoff
        self.device = torch.device('cuda:0')
        self.preprocess()

    def preprocess(self):
        # save activations and get save_paths
        concept_set = "dataset/concept_sets/{}_filtered.txt".format(self.dataset)
        # concept_set = "dataset/concept_sets/deepseek_{}_filtered.txt".format(self.dataset)
        # concept_set = "dataset/concept_sets/gpt4o_{}_filtered.txt".format(self.dataset)
        # concept_set = "dataset/concept_sets/expert_{}_filtered.txt".format(self.dataset)
        # get concept set
        cls_file = LABEL_FILES[self.dataset.split("_")[-1]]
        with open(cls_file, "r") as f:
            classes = f.read().split("\n")
        with open(concept_set) as f:
            concepts = f.read().split("\n")

        d_train = self.dataset + "_train"
        d_val = self.dataset + "_val"
        for d_probe in [d_train, d_val]:
            save_activations(clip_name = self.clip_name, target_name = self.backbone, 
                                    target_layers = [self.feature_layer], d_probe = d_probe,
                                    concept_set = concept_set, batch_size = self.activation_batch_size, 
                                    device = self.device, pool_mode = "avg", save_dir = self.save_dir)

        target_save_name, clip_save_name, text_save_name = get_save_names(self.clip_name, self.backbone, 
                                                self.feature_layer, d_train, concept_set, "avg", self.save_dir)
        val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(self.clip_name, self.backbone,
                                                self.feature_layer, d_val, concept_set, "avg", self.save_dir)
        #load features
        with torch.no_grad():
            self.target_features = torch.load(target_save_name, map_location="cpu").float()
            self.val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        
            image_features = torch.load(clip_save_name, map_location="cpu").float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)

            val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
            val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

            text_features = torch.load(text_save_name, map_location="cpu").float()
            text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            clip_features = image_features @ text_features.T
            # val_clip_features = val_image_features @ text_features.T

            del text_features#, val_clip_features

        #filter concepts not activating highly
        highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

        # for i, concept in enumerate(concepts):
        #     if highest[i]<=self.clip_cutoff:
        #         print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))

        concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>self.clip_cutoff]

        #save memory by recalculating 
        del clip_features
        with torch.no_grad():
            self.text_features = torch.load(text_save_name, map_location="cpu").float()[highest>self.clip_cutoff]
            self.text_features /= torch.norm(self.text_features, dim=1, keepdim=True)
        
            self.clip_features = image_features @ self.text_features.T
            self.val_clip_features = val_image_features @ self.text_features.T
            del image_features
        
        # self.val_clip_features = self.val_clip_features[:, highest>self.clip_cutoff]

        self.train_targets = get_targets_only(d_train)
        self.val_targets = get_targets_only(d_val)
        self.train_targets = torch.LongTensor(self.train_targets)
        self.val_targets = torch.LongTensor(self.val_targets)
        self.num_concepts = len(concepts)
        self.num_class = len(classes)

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.target_features, self.clip_features, self.train_targets)
        self.val_dataset = TensorDataset(self.val_target_features, self.val_clip_features, self.val_targets)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return MultiEpochsDataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=4
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=4
                          )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=4
                          )

class VLMDHyperInterface(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.dataset = data_config.dataset
        self.clip_name = data_config.clip_name
        self.backbone = data_config.backbone
        self.feature_layer = data_config.feature_layer
        self.activation_batch_size = data_config.activation_batch_size
        self.batch_size = data_config.batch_size
        self.save_dir = data_config.save_dir
        self.num_workers = data_config.num_workers
        self.device = torch.device('cuda:0')

    def setup(self, stage=None):
        # get class
        d_train = self.dataset + "_train"
        d_val = self.dataset + "_val"
        train_targets = get_targets_only(d_train)
        val_targets = get_targets_only(d_val)
        train_targets = torch.LongTensor(train_targets)
        val_targets = torch.LongTensor(val_targets)

        cls_file = LABEL_FILES[self.dataset]
        with open(cls_file, "r") as f:
            classes = f.read().split("\n")
        self.num_class = len(classes)

        # ##############################
        # concept_set = "dataset/concept_sets/{}_filtered.txt".format(self.dataset)
        # target_save_name, clip_save_name, text_save_name = get_save_names(self.clip_name, self.backbone, 
        #                                         self.feature_layer, d_train, concept_set, "avg", self.save_dir)
        # val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(self.clip_name, self.backbone,
        #                                         self.feature_layer, d_val, concept_set, "avg", self.save_dir)
        # #load features
        # with torch.no_grad():
        #     self.target_features = torch.load(target_save_name, map_location="cpu").float()
        #     self.val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        # ##############################

        # if self.dataset == 'imagenet':
        #     train_outs_c = torch.load(f'{self.save_dir}{self.dataset}_train_outs_c_1.pt')
        #     for i in range(1):
        #         train_outs_c_slice = torch.load(f'{self.save_dir}{self.dataset}_train_outs_c_{i+2}.pt')
        #         train_outs_c = torch.cat((train_outs_c, train_outs_c_slice), dim=0)
        # else:
        
        try:
            train_outs_c = torch.load(f'{self.save_dir}{self.dataset}_train_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}.pt')
        except:
            train_outs_c = load_tensors_incrementally(f'{self.save_dir}{self.dataset}_train_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}_batches/')
            # val_outs_c = load_tensors_incrementally(f'{self.save_dir}{self.dataset}_test_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}_batches/')
        val_outs_c = torch.load(f'{self.save_dir}{self.dataset}_test_outs_c_{self.clip_name.replace("/", "-")}_{self.backbone.replace("/", "-")}.pt')
        train_mean = torch.mean(train_outs_c, dim=0, keepdim=True)
        train_std = torch.std(train_outs_c, dim=0, keepdim=True)

        train_outs_c -= train_mean
        train_outs_c /= train_std
        val_outs_c -= train_mean
        val_outs_c /= train_std
        self.train_dataset = TensorDataset(train_outs_c, train_targets)
        self.val_dataset = TensorDataset(val_outs_c, val_targets)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return MultiEpochsDataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=16, 
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=4
                          )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                        #   prefetch_factor=4
                          )