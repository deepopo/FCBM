import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sparsemax

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=100):
        super(ThreeLayerMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First layer
            nn.ReLU(),                         # Activation
            nn.Linear(hidden_dim, hidden_dim), # Second layer
            nn.ReLU(),                         # Activation
            nn.Linear(hidden_dim, output_dim)  # Output layer
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class VLMConceptFCBM(nn.Module):
    def __init__(self, target_features_dim, num_concepts):
        super(VLMConceptFCBM, self).__init__()
        self.proj_layer = torch.nn.Linear(in_features=target_features_dim, out_features=num_concepts, bias=False)
    def forward(self, target_features):
        outs_c = self.proj_layer(target_features)  # [batch_size, num_concepts]
        return outs_c

class VLMHyperFCBM(torch.nn.Module):
    def __init__(self, text_features_dim, num_class, init_scale_factor):
        super(VLMHyperFCBM, self).__init__()
        self.hypernet = ThreeLayerMLP(input_dim=text_features_dim, hidden_dim=4096, output_dim=num_class)

        # self.hypernet = nn.Linear(text_features_dim, num_class)  # [num_concepts, 100]
        self.scale_factor = nn.Parameter(torch.tensor(init_scale_factor))  # learnable scale factor
        self.temperature = nn.Parameter(torch.tensor(10.0))  # learnable temperature

    def forward(self, outs_c, text_features, sparse=True, train_concept_prop=1., test_zero_shot=False):
        '''
            outs_c: [batch_size, num_concepts]
            text_features: [num_concepts, text_features_dim]
            sparse: whether to use sparsemax
            train_concept_prop: proportion of training concepts
            test_zero_shot: whether to test zero-shot learning on untrained concepts
        '''
        self.sparse = sparse
        num_concepts = text_features.size(0)
        train_num_concepts = int(num_concepts * train_concept_prop)
        assert train_num_concepts > 1
        if not sparse:
            if not test_zero_shot:
                outs_c = outs_c[:, :train_num_concepts]
                text_features = text_features[:train_num_concepts]
                weights = self.hypernet(text_features) * self.scale_factor  # [num_concepts, 100]
            else:
                mean_train = torch.mean(text_features[:train_num_concepts], dim=0)
                std_train = torch.std(text_features[:train_num_concepts], dim=0, unbiased=False)
                mean_test = torch.mean(text_features[train_num_concepts:], dim=0)
                std_test = torch.std(text_features[train_num_concepts:], dim=0, unbiased=False)
                # update text_features
                text_features_test = (text_features[train_num_concepts:] - mean_test) / std_test * std_train + mean_train
                text_features[train_num_concepts:] = text_features_test
                outs_c = outs_c[:, train_num_concepts:]

                weights = self.hypernet(text_features) * self.scale_factor  # [num_concepts, 100]
                mean_train_weights = torch.mean(weights[:train_num_concepts], dim=0)
                std_train_weights = torch.std(weights[:train_num_concepts], dim=0, unbiased=False)
                mean_test_weights = torch.mean(weights[train_num_concepts:], dim=0)
                std_test_weights = torch.std(weights[train_num_concepts:], dim=0, unbiased=False)
                weights = (weights[train_num_concepts:] - mean_test_weights) / std_test_weights * std_train_weights + mean_train_weights
        else:
            if not test_zero_shot:
                outs_c = outs_c[:, :train_num_concepts]
                text_features = text_features[:train_num_concepts]
                weights = self.hypernet(text_features)  # [num_concepts, 100]
                sign_weights = torch.sign(weights)
                weights = sparsemax(weights.abs(), dim=0, temperature=self.temperature) * self.scale_factor
                weights = weights * sign_weights
            else:
                mean_train = torch.mean(text_features[:train_num_concepts], dim=0)
                std_train = torch.std(text_features[:train_num_concepts], dim=0, unbiased=False)
                mean_test = torch.mean(text_features[train_num_concepts:], dim=0)
                std_test = torch.std(text_features[train_num_concepts:], dim=0, unbiased=False)
                # update text_features
                text_features_test = (text_features[train_num_concepts:] - mean_test) / std_test * std_train + mean_train
                text_features[train_num_concepts:] = text_features_test
                outs_c = outs_c[:, train_num_concepts:]

                weights = self.hypernet(text_features)  # [num_concepts, 100]
                mean_train_weights = torch.mean(weights[:train_num_concepts], dim=0)
                std_train_weights = torch.std(weights[:train_num_concepts], dim=0, unbiased=False)
                mean_test_weights = torch.mean(weights[train_num_concepts:], dim=0)
                std_test_weights = torch.std(weights[train_num_concepts:], dim=0, unbiased=False)
                weights = (weights[train_num_concepts:] - mean_test_weights) / std_test_weights * std_train_weights + mean_train_weights
                sign_weights = torch.sign(weights)
                weights = sparsemax(weights.abs(), dim=0, temperature=self.temperature) * self.scale_factor
                weights = weights * sign_weights

        self.avg_act_concepts = (weights.abs() > 1e-5).sum().item()/(weights.size(1))
        outs_y = outs_c.mm(weights)  # [batch_size, num_classes]
        return outs_y