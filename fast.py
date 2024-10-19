import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F
import glob


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 64)

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.stage1(x)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)

        # return features
        return x2, x3, x4, x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()

        # self.reduce2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        # self.reduce3 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        # self.reduce4 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        # self.reduce5 = nn.Conv2d(512, 128, 3, padding=1, bias=False)
        self.reduce5 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.reduce4 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.reduce3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        
        self.smooth4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
    def upsample_cat(self, p2, p3, p4, p5):
        size = p2.size()[2:]
        # p3 = F.interpolate(p3, size=size, mode="bilinear")
        # p4 = F.interpolate(p4, size=size, mode="bilinear")
        # p5 = F.interpolate(p5, size=size, mode="bilinear")
        p3 = F.interpolate(p3, size=size)
        p4 = F.interpolate(p4, size=size)
        p5 = F.interpolate(p5, size=size)
        return torch.cat([p2, p3, p4, p5], dim=1)

    def forward(self, c2, c3, c4, c5):
        # TODO smooth layers
        p5 = self.reduce5(c5)
        p4 = F.interpolate(p5, size=c4.size()[2:]) + self.reduce4(c4)
        p4 = self.smooth4(p4)
        p3 = F.interpolate(p4, size=c3.size()[2:]) + self.reduce3(c3)
        p3 = self.smooth3(p3)
        p2 = F.interpolate(p3, size=c2.size()[2:]) + self.reduce2(c2)
        p2 = self.smooth2(p2)

        p = self.upsample_cat(p2, p3, p4, p5)
        p = self.conv(p)
        return p


loss_kernel_fn = nn.BCELoss()
loss_text_fn = nn.BCELoss()


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x).squeeze(1)

    def get_unified_focal_loss_asym(self, pred, true):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5

        pt = true * pred + (1 - true) * (1 - pred)
        L_maF = -delta * true * torch.log(pt + 1e-8) - (1 - delta) * torch.pow(
            pred + 1e-8, gamma
        ) * torch.log(pt + 1e-8)
        L_maF = -delta * pred * torch.log(pt + 1e-8)
        L_maF = L_maF.mean()

        mTI = (pred * true).sum() / (
            pred * true
            + delta * (pred * (1 - true))
            + (1 - delta) * ((1 - pred) * true)
        ).sum()
        L_maFT = (1 - mTI) + torch.pow(1 - mTI + 1e-8, 1 - gamma)

        return lmd * L_maF + (1 - lmd) * L_maFT

    def get_unified_focal_loss_sym(self, pred, true, edge_weight):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5

        pt = true * pred + (1 - true) * (1 - pred)
        L_mF = (
            delta
            * (1 - pt + 1e-8).pow(gamma)
            * F.binary_cross_entropy(pred, true, reduction="none")
        )
        L_mF = L_mF * edge_weight
        L_mF = L_mF.mean()

        mTI = (edge_weight * pred * true).sum() / (
            edge_weight
            * (
                pred * true
                + delta * (pred * (1 - true))
                + (1 - delta) * ((1 - pred) * true)
            )
        ).sum()
        L_mFT = (1 - mTI).pow(gamma)

        return lmd * L_mF + (1 - lmd) * L_mFT

    def loss(self, out, gt_kernels, gt_texts, edge_weight):
        out = out.squeeze(1)

        loss_kernel_old = loss_kernel_fn(out, gt_kernels)
        # loss_kernel = 1 - (2 * (out * gt_kernels).sum()) / (torch.pow(out, 2).sum() + torch.pow(gt_kernels, 2).sum())
        # loss_kernel = self.get_unified_focal_loss_asym(out, gt_kernels)
        loss_kernel = self.get_unified_focal_loss_sym(out, gt_kernels, edge_weight)

        pred_text = F.max_pool2d(out, 9, stride=1, padding=4)
        loss_text_old = loss_text = loss_text_fn(pred_text, gt_texts)
        # loss_text = 1 - (2 * (pred_text * gt_texts).sum()) / (torch.pow(pred_text, 2).sum() + torch.pow(gt_texts, 2).sum())
        # loss_text = self.get_unified_focal_loss_asym(pred_text, gt_texts)
        loss_text = self.get_unified_focal_loss_sym(pred_text, gt_texts, torch.ones_like(edge_weight))

        return loss_kernel + loss_text, loss_kernel_old + 0.5 * loss_text_old


class FAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)
        x = self.neck(c2, c3, c4, c5)
        x = self.head(x)

        return x