# model.py - Neural network architectures for Roblox AI

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch)
        self.conv2 = DepthwiseSeparableConv(ch, ch)
        self.bn    = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))) + x, inplace=True)


class GameEncoder(nn.Module):
    def __init__(self, in_channels=config.IN_CHANNELS, hidden=config.HIDDEN_DIM):
        super().__init__()
        self.stem   = nn.Sequential(ConvBNReLU(in_channels, 32, k=3, s=2, p=1), ConvBNReLU(32, 64, k=3, s=2, p=1))
        self.stage1 = nn.Sequential(DepthwiseSeparableConv(64, 128, stride=2), ResBlock(128), ResBlock(128))
        self.stage2 = nn.Sequential(DepthwiseSeparableConv(128, 256, stride=2), ResBlock(256), ResBlock(256))
        self.stage3 = nn.Sequential(DepthwiseSeparableConv(256, 512, stride=2), ResBlock(512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.proj   = nn.Sequential(nn.Flatten(), nn.Linear(512, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(config.DROPOUT))
    def forward(self, x):
        return self.proj(self.pool(self.stage3(self.stage2(self.stage1(self.stem(x))))))
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GameAI(nn.Module):
    """Main behavioural cloning model with 4 action heads."""
    def __init__(self):
        super().__init__()
        self.encoder   = GameEncoder()
        h = config.HIDDEN_DIM
        self.shared    = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(config.DROPOUT))
        self.head_keys    = nn.Linear(h, len(config.KEY_ACTIONS))
        self.head_mouse_x = nn.Linear(h, config.MOUSE_BINS)
        self.head_mouse_y = nn.Linear(h, config.MOUSE_BINS)
        self.head_clicks  = nn.Linear(h, len(config.MOUSE_BUTTONS))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        feat = self.shared(self.encoder(x))
        return {"key_logits": self.head_keys(feat), "mouse_x_logits": self.head_mouse_x(feat),
                "mouse_y_logits": self.head_mouse_y(feat), "click_logits": self.head_clicks(feat)}
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InverseDynamicsNet(nn.Module):
    """Pretraining model - predicts frame transition magnitude."""
    def __init__(self):
        super().__init__()
        self.encoder = GameEncoder(in_channels=6, hidden=config.HIDDEN_DIM)
        self.head = nn.Sequential(nn.Linear(config.HIDDEN_DIM, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): return self.head(self.encoder(x)).squeeze(-1)
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
