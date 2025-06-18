# cross_kd2.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCAProjector(nn.Module):
    """Partially Cross-Attention Projector"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_c = out_channels or in_channels
        self.query_conv = nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1)
        self.key_conv   = nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        B, _, H, W = x.shape
        # project
        q = self.query_conv(x).flatten(2).permute(0, 2, 1)  # [B, N, C']
        k = self.key_conv(x).flatten(2)                     # [B, C', N]
        v = self.value_conv(x).flatten(2).permute(0, 2, 1)  # [B, N, C']
        # attention
        attn = torch.bmm(q, k) / math.sqrt(k.size(1))       # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        # context
        ctx = torch.bmm(attn, v)                            # [B, N, C']
        ctx = ctx.permute(0, 2, 1).view(B, -1, H, W)
        return attn, ctx

class GLProjector(nn.Module):
    """Group-wise Linear Projector"""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.groups = groups
        ic = (in_channels // groups) * groups
        oc = (out_channels // groups) * groups
        self.group_linears = nn.ModuleList([
            nn.Conv2d(ic // groups, oc // groups, kernel_size=1)
            for _ in range(groups)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        if C % self.groups != 0:
            pad = (self.groups - (C % self.groups))
            x = F.pad(x, (0,0,0,0,0,pad))
        chunks = torch.chunk(x, self.groups, dim=1)
        outs = [self.group_linears[i](chunks[i]) for i in range(self.groups)]
        return torch.cat(outs, dim=1)

class MultiViewGenerator(nn.Module):
    """Generate multiple views of an input for robust training"""
    def __init__(self, num_views=2):
        super().__init__()
        self.num_views = num_views

    def forward(self, x):
        views = [x]
        if self.num_views < 2:
            return views
        B, C, H, W = x.shape
        side = int(0.8 * min(H, W))
        starts = torch.randint(0, min(H, W) - side + 1, (B, 2), device=x.device)
        crops = []
        for b in range(B):
            sh, sw = starts[b]
            crop = x[b:b+1, :, sh:sh+side, sw:sw+side]
            crops.append(F.interpolate(crop, (H, W),
                                       mode='bilinear', align_corners=False))
        views.append(torch.cat(crops, dim=0))
        return views

class Discriminator(nn.Module):
    """Discriminator for robust training"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W] or [B, C]
        if x.dim() > 2:
            x = x.mean(dim=[2,3])
        return self.net(x)

class CrossArchitectureKD(nn.Module):
    """Cross-Architecture Knowledge Distillation"""
    def __init__(self, teacher, student, feature_dim, num_views=2, lambda_robust=1.0):
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student
        self.teacher_dim   = feature_dim
        self.lambda_robust = lambda_robust

        # Infer actual student feature dim with a dummy forward:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224,
                                device=next(student.parameters()).device)
            _, feats = student(dummy, return_features=True)
            self.student_dim = feats.shape[1]

        # Projectors
        self.pca = PCAProjector(self.student_dim)
        self.gl  = GLProjector(self.student_dim, self.teacher_dim)
        # Identity match for teacher→GL target
        self.channel_matcher = nn.Identity()
        self.mv  = MultiViewGenerator(num_views)
        self.disc= Discriminator(self.teacher_dim)

    def forward(self, x):
        return self.student(x)

    def calculate_kd_loss(self, x):
        views = self.mv(x)
        device = x.device
        pca_loss = torch.tensor(0., device=device)
        gl_loss  = torch.tensor(0., device=device)
        rob_loss = torch.tensor(0., device=device)

        for v in views:
            # 1) teacher features (no grad)
            with torch.no_grad():
                _, t_feat = self.teacher(v, return_features=True)
            # 2) student features
            _, s_feat = self.student(v, return_features=True)

            # PCA branch
            s_att, _ = self.pca(s_feat)  # [B, N, N]
            B, _, H, W = s_feat.shape

            # get teacher spatial
            if t_feat.dim() == 4:
                t_sp = t_feat
            else:
                B2, seq, d = t_feat.shape
                tokens = t_feat[:, 1:, :]  # drop CLS
                side   = int(math.sqrt(tokens.size(1)))
                t_sp   = tokens.permute(0,2,1).view(B2, d, side, side)
            t_sp = F.interpolate(t_sp, size=(H,W), mode='bilinear', align_corners=False)

            q = t_sp.flatten(2).permute(0,2,1)
            k = t_sp.flatten(2)
            t_att = torch.bmm(q, k) / math.sqrt(self.teacher_dim)
            t_att = F.softmax(t_att, dim=-1)
            pca_loss += F.kl_div(F.log_softmax(s_att, dim=-1), t_att, reduction='batchmean')

            # GL branch (now shapes match)
            s_gl = self.gl(s_feat)            # [B, teacher_dim, H, W]
            t_gl = self.channel_matcher(t_sp)  # Identity: [B, teacher_dim, H, W]
            gl_loss += F.mse_loss(s_gl, t_gl)

            # Robust/adversarial branch
            d_s = self.disc(s_gl)
            rob_loss += F.binary_cross_entropy_with_logits(
                d_s, torch.ones_like(d_s), reduction='mean'
            )

        # Average over views
        n = len(views)
        pca_loss /= n
        gl_loss  /= n
        rob_loss /= n

        total = pca_loss + self.lambda_robust * gl_loss + rob_loss
        return total, pca_loss, gl_loss, rob_loss
