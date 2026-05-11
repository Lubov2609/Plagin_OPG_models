import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


class DenseNetModel(nn.Module):
    """
    Multi-task DenseNet121 used by the current dental_cnn_project checkpoint:
      0. caries binary logit     -> [B, 1]
      1. pulpitis binary logit   -> [B, 1]
      2. erupted_8 binary logit  -> [B, 1]
      3. retained_8 logits 0..4  -> [B, 5]
    """

    def __init__(
        self,
        num_classes=5,
        num_binary_tasks=3,
        num_multiclass_tasks=1,
        dropout_binary=0.35,
        dropout_multi=0.30,
        unfreeze_denseblock4=False,
        unfreeze_denseblock3=False,
        freeze_bn_stats=True,
    ):
        super().__init__()

        self.backbone = densenet121(weights=None, memory_efficient=True)
        self.freeze_bn_stats = freeze_bn_stats
        self.num_binary_tasks = int(num_binary_tasks)
        self.num_multiclass_tasks = int(num_multiclass_tasks)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_denseblock4:
            for param in self.backbone.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.backbone.features.norm5.parameters():
                param.requires_grad = True

        if unfreeze_denseblock3:
            for param in self.backbone.features.denseblock3.parameters():
                param.requires_grad = True
            for param in self.backbone.features.transition3.parameters():
                param.requires_grad = True

        self.pool_mid = nn.AdaptiveAvgPool2d(1)
        self.pool_deep = nn.AdaptiveAvgPool2d(1)

        self.binary_tower = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_binary),
            nn.Linear(192, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )

        self.multi_tower = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_multi),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )

        self.binary_heads = nn.ModuleList([
            nn.Linear(64, 1)
            for _ in range(self.num_binary_tasks)
        ])
        self.multiclass_heads = nn.ModuleList([
            nn.Linear(64, num_classes)
            for _ in range(self.num_multiclass_tasks)
        ])

        self._init_new_layers()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _init_new_layers(self):
        for module in [self.binary_tower, self.multi_tower, self.binary_heads, self.multiclass_heads]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _prepare_input(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def _forward_features(self, x):
        features = self.backbone.features

        x = features.conv0(x)
        x = features.norm0(x)
        x = features.relu0(x)
        x = features.pool0(x)

        x = features.denseblock1(x)
        x = features.transition1(x)
        x = features.denseblock2(x)
        x = features.transition2(x)

        mid = features.denseblock3(x)
        x = features.transition3(mid)
        x = features.denseblock4(x)
        x = features.norm5(x)
        deep = F.relu(x, inplace=False)

        return mid, deep

    def forward(self, x):
        x = self._prepare_input(x)
        mid, deep = self._forward_features(x)

        mid_feat = self.pool_mid(mid).flatten(1)
        deep_feat = self.pool_deep(deep).flatten(1)

        binary_feat = self.binary_tower(torch.cat([mid_feat, deep_feat], dim=1))
        multi_feat = self.multi_tower(deep_feat)

        return [head(binary_feat) for head in self.binary_heads] + [
            head(multi_feat) for head in self.multiclass_heads
        ]

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    if module.weight is not None:
                        module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False
        return self

    def get_parameter_groups(self, lr_backbone=3e-6, lr_head=2e-4):
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head})
        return groups
