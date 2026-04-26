import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # grayscale input
        old_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.backbone.features.conv0 = new_conv

        # замораживаем backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # размораживаем последние блоки
        for param in self.backbone.features.denseblock3.parameters():
            param.requires_grad = True

        for param in self.backbone.features.denseblock4.parameters():
            param.requires_grad = True

        for param in self.backbone.features.norm5.parameters():
            param.requires_grad = True

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        # shared head
        self.shared = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 2 бинарные головы
        self.binary_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(2)
        ])

        # 2 мультиклассовые головы
        self.multiclass_heads = nn.ModuleList([
            nn.Linear(128, 5) for _ in range(2)
        ])

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        binary_outputs = [head(x) for head in self.binary_heads]
        multiclass_outputs = [head(x) for head in self.multiclass_heads]

        return binary_outputs + multiclass_outputs