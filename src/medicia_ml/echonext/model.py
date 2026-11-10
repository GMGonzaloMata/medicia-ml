"""Arquitectura de EchoNext-Mini: una ResNet 1D con variables tabulares.

Escrita de forma independiente a partir de la descripción de la arquitectura. Los nombres de los módulos
son los que hacen coincidir el state_dict de los pesos publicados; se carga con strict=True, así que una
diferencia de nombres o de formas falla de inmediato. Solo se usa para predecir (modo eval).
"""

import torch
from torch import nn

from medicia_ml.echonext.contract import FINDINGS, TABULAR_FEATURES, WAVEFORM_LEADS

FILTER_SIZE = 16
_STAGES = (3, 4, 6, 3)
_KERNEL = 7
_PADDING = 3


class BasicBlock1d(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, _KERNEL, stride=stride, padding=_PADDING, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(planes, planes, _KERNEL, stride=1, padding=_PADDING, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class EchoNextModel(nn.Module):
    def __init__(
        self,
        num_tabular: int = len(TABULAR_FEATURES),
        num_classes: int = len(FINDINGS),
        filter_size: int = FILTER_SIZE,
    ):
        super().__init__()
        self._inplanes = filter_size
        self.conv1 = nn.Conv1d(WAVEFORM_LEADS, filter_size, 15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(filter_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(filter_size, _STAGES[0])
        self.layer2 = self._make_layer(2 * filter_size, _STAGES[1], stride=2)
        self.layer3 = self._make_layer(4 * filter_size, _STAGES[2], stride=2)
        self.layer4 = self._make_layer(8 * filter_size, _STAGES[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(8 * filter_size * 2 + num_tabular, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self._inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1d(self._inplanes, planes, stride, downsample)]
        self._inplanes = planes
        layers.extend(BasicBlock1d(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """waveform: (N, 1, 2500, 12); tabular: (N, 7). Devuelve los logits (N, 12)."""
        x = torch.transpose(waveform, 2, 3).squeeze(1)  # (N, 12, 2500)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.cat((self.adaptiveavgpool(x), self.adaptivemaxpool(x)), dim=1)
        x = x.view(x.size(0), -1)
        return self.output(torch.cat((x, tabular), dim=1))
