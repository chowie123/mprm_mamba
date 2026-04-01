# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# MONAI
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

# Mamba (pip install mamba-ssm or your local build)
from mamba_ssm import Mamba


# ------------------------------
# 残差缩放 / DropPath
# ------------------------------
class ResidualScale(nn.Module):
    def __init__(self, init_value: float = 1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep).div(keep)
        return x * mask


# ------------------------------
# Norm helpers
# ------------------------------
class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last/channels_first for 3D."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ("channels_last", "channels_first")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# ------------------------------
# SE3D / FRM / GSC / MlpChannel
# ------------------------------
class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w


class DepthwiseConv3d(nn.Conv3d):
    def __init__(self, channels: int, k: int = 5, stride: int = 1):
        super().__init__(channels, channels, kernel_size=k, stride=stride,
                         padding=k // 2, groups=channels, bias=True)


class FRM(nn.Module):
    def __init__(self, channels: int, use_instance_norm: bool = True, dw_k: int = 5,
                 res_init: float = 1e-3, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.InstanceNorm3d(channels) if use_instance_norm else LayerNorm(
            channels, data_format="channels_first"
        )
        self.lin1 = nn.Conv3d(channels, channels, 1)
        self.se = SE3D(channels)
        self.half = max(1, channels // 2)
        self.dw = DepthwiseConv3d(self.half, k=dw_k)
        self.pw = nn.Conv3d(self.half, self.half, 1)
        self.sig = nn.Sigmoid()
        self.lin2 = nn.Conv3d(self.half, channels, 1)
        self.res_scale = ResidualScale(res_init)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.lin1(self.norm(x))     # (B,C,D,H,W)
        u = self.se(u)
        c1 = self.half
        u1, u2 = torch.split(u, [c1, u.shape[1] - c1], dim=1)
        g = self.sig(self.pw(self.dw(u2)))
        v = u1 * g
        y = self.lin2(v)
        return x + self.res_scale(self.dp(y))


class GSC(nn.Module):
    
    def __init__(self, in_channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU(inplace=True)
        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU(inplace=True)
        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU(inplace=True)
        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU(inplace=True)
        self.res_scale = ResidualScale(1e-3)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = x
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        x2 = self.nonliner3(self.norm3(self.proj3(x)))
        y = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
        return x_residual + self.res_scale(self.dp(y))


class MlpChannel(nn.Module):
    
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------
# 三正交方向门控（Axial / Coronal / Sagittal）
# ------------------------------
class CrossDirectionalGating3(nn.Module):
    """
    A_i = σ(Wm ReLU(Wg GAP(concat(others)) + Ws Zi)),  Z'i = A_i ⊙ Zi
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.Wg = nn.Conv3d(2 * channels, channels, 1)
        self.Ws = nn.Conv3d(channels, channels, 1)
        self.Wm = nn.Conv3d(channels, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def _gate_one(self, Zi: torch.Tensor, others: List[torch.Tensor]) -> torch.Tensor:
        Gi = torch.cat(others, dim=1)
        ctx = self.Wg(self.pool(Gi))
        self_term = self.Ws(Zi)
        A = self.sig(self.Wm(self.act(ctx + self_term)))
        return A * Zi

    def forward(self, Za: torch.Tensor, Zc: torch.Tensor, Zs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ya = self._gate_one(Za, [Zc, Zs])
        Yc = self._gate_one(Zc, [Za, Zs])
        Ys = self._gate_one(Zs, [Za, Zc])
        return Ya, Yc, Ys


# ------------------------------
# 每模态内部：三正交方向 Mamba
# ------------------------------
class DirectionalMamba3D(nn.Module):
    """
    - 对每个方向：切片序列化（GAP -> token）→ LayerNorm → Mamba → 回填体素
    - 三方向门控 + 3C->C 投影 + IN + 残差缩放
    - 输出维度与输入一致：C -> C
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 pool_reduce: int = 1, up_project: bool = True,
                 res_init: float = 1e-3, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        # 三个方向的 Mamba
        self.mamba_ax = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_co = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_sa = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ln_ax = nn.LayerNorm(dim)
        self.ln_co = nn.LayerNorm(dim)
        self.ln_sa = nn.LayerNorm(dim)

        # 可选 token 维降维
        self.pool_reduce = pool_reduce
        if pool_reduce > 1:
            self.reduce = nn.Conv1d(dim, dim // pool_reduce, kernel_size=1, bias=False)
            self.expand = nn.Conv1d(dim // pool_reduce, dim, kernel_size=1, bias=False)
        else:
            self.reduce = self.expand = nn.Identity()

        # 回填体素后的轻量上投影
        self.up_ax = nn.Conv3d(dim, dim, 1) if up_project else nn.Identity()
        self.up_co = nn.Conv3d(dim, dim, 1) if up_project else nn.Identity()
        self.up_sa = nn.Conv3d(dim, dim, 1) if up_project else nn.Identity()

        self.cdg3 = CrossDirectionalGating3(dim)

        # 稳定化输出：3C->C，归一化与残差缩放
        self.proj_out = nn.Conv3d(3 * dim, dim, 1)
        self.out_norm = nn.InstanceNorm3d(dim)
        self.res_scale = ResidualScale(res_init)
        self.dp = DropPath(drop_path)

    @staticmethod
    def _seq_pool_axis(x, axis: int):
        # x: (B,C,D,H,W)；沿 axis 方向保留长度，其余两个维做 GAP，得到 (B,C,L)
        B, C, D, H, W = x.shape
        if axis == 2:      # axial -> L = D
            seq = F.adaptive_avg_pool3d(x, output_size=(D, 1, 1))[:, :, :, 0, 0]
        elif axis == 3:    # coronal -> L = H
            seq = F.adaptive_avg_pool3d(x, output_size=(1, H, 1))[:, :, 0, :, 0]
        else:              # sagittal -> L = W
            seq = F.adaptive_avg_pool3d(x, output_size=(1, 1, W))[:, :, 0, 0, :]
        return seq  # (B,C,L)

    @staticmethod
    def _seq_back_to_volume(tokens: torch.Tensor, x_shape: Tuple[int, int, int, int, int], axis: int) -> torch.Tensor:
        # tokens: (B,C,L) -> (B,C,D,H,W) 通过正交平面广播
        B, C, L = tokens.shape
        _, _, D, H, W = x_shape
        if axis == 2:      # axial
            vol = tokens.view(B, C, L, 1, 1).expand(B, C, D, H, W)
        elif axis == 3:    # coronal
            vol = tokens.view(B, C, 1, L, 1).expand(B, C, D, H, W)
        else:              # sagittal
            vol = tokens.view(B, C, 1, 1, L).expand(B, C, D, H, W)
        return vol

    def _run_one_axis(self, x: torch.Tensor, axis: int, mamba: nn.Module, ln: nn.Module, up: nn.Module) -> torch.Tensor:
        seq = self._seq_pool_axis(x, axis).transpose(1, 2)                # (B,L,C)
        seq = self.reduce(seq.transpose(1, 2)).transpose(1, 2)            # (B,L,C') or Identity
        seq = ln(seq)
        seq = mamba(seq)                                                  # (B,L,C')
        seq = self.expand(seq.transpose(1, 2)).transpose(1, 2)            # (B,L,C)
        vol = self._seq_back_to_volume(seq.transpose(1, 2), x.shape, axis)  # (B,C,D,H,W)
        vol = up(vol)
        return vol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Za = self._run_one_axis(x, axis=2, mamba=self.mamba_ax, ln=self.ln_ax, up=self.up_ax)
        Zc = self._run_one_axis(x, axis=3, mamba=self.mamba_co, ln=self.ln_co, up=self.up_co)
        Zs = self._run_one_axis(x, axis=4, mamba=self.mamba_sa, ln=self.ln_sa, up=self.up_sa)
        Ya, Yc, Ys = self.cdg3(Za, Zc, Zs)
        y = torch.cat([Ya, Yc, Ys], dim=1)      # (B,3C,...)
        y = self.proj_out(y)                    # (B,C,...)
        y = self.out_norm(y)
        return x + self.res_scale(self.dp(y))   # 稳定残差


# ------------------------------
# 3D Patch Tokenizer / Unpatchifier
# ------------------------------
class PatchTokenizer3D(nn.Module):
    """
    将 (B,C,D,H,W) 切成 3D 小块并展平为 token：
      tokens: (B, N, C*Pz*Py*Px)
    N = (D'/Pz)*(H'/Py)*(W'/Px)；对 D/H/W 做必要 padding
    """
    def __init__(self, patch_size: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.pz, self.py, self.px = patch_size

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor, pz: int, py: int, px: int
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
        B, C, D, H, W = x.shape
        Dz = (pz - D % pz) % pz
        Hy = (py - H % py) % py
        Wx = (px - W % px) % px
        if Dz or Hy or Wx:
            x = F.pad(x, (0, Wx, 0, Hy, 0, Dz))
        return x, (D, H, W), (D + Dz, H + Hy, W + Wx)

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, C, D, H, W = x.shape
        x, orig, padded = self._pad_to_multiple(x, self.pz, self.py, self.px)
        B, C, Dp, Hp, Wp = x.shape
        gz, gy, gx = Dp // self.pz, Hp // self.py, Wp // self.px
        x = x.view(B, C, gz, self.pz, gy, self.py, gx, self.px)           # (B,C,gz,pz,gy,py,gx,px)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()                 # (B,gz,gy,gx,C,pz,py,px)
        N = gz * gy * gx
        tokens = x.view(B, N, C * self.pz * self.py * self.px)            # (B,N,C*P)
        info = {
            "orig": orig,
            "padded": (Dp, Hp, Wp),
            "grid": (gz, gy, gx),
            "patch": (self.pz, self.py, self.px),
            "C": C,
        }
        return tokens, info

    @staticmethod
    def unpatchify(tokens: torch.Tensor, info: dict, out_channels: int) -> torch.Tensor:
        """
        输入 tokens: (B, N, E)，其中 E == out_channels
        将每个 token 的通道向量在对应 (pz,py,px) 小块中做重复填充，
        得到 (B, out_channels, Dp, Hp, Wp)
        """
        B, N, E = tokens.shape
        gz, gy, gx = info["grid"]
        pz, py, px = info["patch"]
        Dp, Hp, Wp = info["padded"]

        assert E == out_channels, f"unpatchify expects E==out_channels, got E={E}, out={out_channels}"
        assert N == gz * gy * gx, f"N grid mismatch: N={N}, grid={gz}x{gy}x{gx}"

        x = tokens.view(B, gz, gy, gx, E).permute(0, 4, 1, 2, 3).contiguous()  # (B,E,gz,gy,gx)
        x = (
            x.unsqueeze(3).unsqueeze(5).unsqueeze(7)                           # (B,E,gz,1,gy,1,gx,1)
             .expand(B, E, gz, pz, gy, py, gx, px)                             # (B,E,gz,pz,gy,py,gx,px)
        )
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()                     # (B,E,gz,gy,gx,pz,py,px)
        x = x.view(B, E, Dp, Hp, Wp)                                           # (B,E,Dp,Hp,Wp)
        return x

    @staticmethod
    def crop_to_orig(x: torch.Tensor, orig: Tuple[int, int, int]) -> torch.Tensor:
        D, H, W = orig
        return x[..., :D, :H, :W]


# ------------------------------
# 条件化 Mamba（FiLM）
# ------------------------------
class ConditionalMamba1D(nn.Module):
    """
    用模态描述符 desc 产生 FiLM 参数 (gamma,beta) 对序列做条件化，再送入 Mamba。
    seq: (B, L, E)
    """
    def __init__(self, embed_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta  = nn.Linear(embed_dim, embed_dim)
        self.mamba = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, seq: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        # seq: (B,L,E), desc: (B,E)
        s = self.ln(seq)
        g = self.gamma(desc).unsqueeze(1)  # (B,1,E)
        b = self.beta(desc).unsqueeze(1)   # (B,1,E)
        s = g * s + b
        h = self.mamba(s)                  # (B,L,E)
        return h


# ------------------------------
# MPRM：跨模态 C 矩阵读出 + α 权重 + 旁路重要性门控
# ------------------------------
class MPRM_Perspective(nn.Module):
    """
    1) 四模态各自 3D patchify -> (B,N,E_in) -> 线性到 embed_dim
    2) desc_i（token 平均）作为全局描述符
    3) 条件化 Mamba：h_i = Mamba(FiLM(seq_i, desc_i))
    4) 相似度矩阵 S(desc) + masked softmax -> α_ij
    5) 跨模态读出：y_i = Σ_j α_ij * (h_i @ C_j) + D_i x_i
       - C_j: 低秩 U_j V_j^T（由 desc_j 生成）
       - D_i: 逐 token 线性校正
    6) unpatchify -> (B, E, Db, Hb, Wb)
    7) 旁路重要性门控： Ẏ_i = IN(Y_i)，M_i = SiLU(U_i)，Z_i = M_i ⊙ Ẏ_i
       - U_i = Conv3D(1x1)(Conv3D(1x1)(IN(x_i)))，输出为 E 通道或 1 通道（由 gate_mode 决定）
    8) 汇总四模态：逐像素求和得到 fused
    """
    def __init__(self, in_ch_each: int = 1, embed_dim: int = 64, n_mods: int = 4,
                 patch_size: Tuple[int, int, int] = (2, 2, 2), lowrank: int = 32,
                 gate_mode: str = "channel"):
        super().__init__()
        assert gate_mode in ("channel", "spatial")
        self.n = n_mods
        self.embed_dim = embed_dim
        self.tokenizer = PatchTokenizer3D(patch_size)
        self.gate_mode = gate_mode

        input_token_dim = in_ch_each * patch_size[0] * patch_size[1] * patch_size[2]
        self.in_proj = nn.ModuleList([nn.Linear(input_token_dim, embed_dim) for _ in range(n_mods)])
        self.cond_mamba = nn.ModuleList([ConditionalMamba1D(embed_dim) for _ in range(n_mods)])

        # 低秩生成 C_j
        self.readout_U = nn.ModuleList([nn.Linear(embed_dim, embed_dim * lowrank) for _ in range(n_mods)])
        self.readout_V = nn.ModuleList([nn.Linear(embed_dim, embed_dim * lowrank) for _ in range(n_mods)])

        # D_i: 逐 token 仿射
        self.D_lin = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_mods)])

        # 输出归一化
        self.out_norm = nn.InstanceNorm3d(embed_dim, affine=True)

        # 旁路：IN -> 1x1 -> 1x1（输出通道=E 或 1）
        gate_out_ch = embed_dim if gate_mode == "channel" else 1
        self.bypass_norm = nn.ModuleList([nn.InstanceNorm3d(in_ch_each, affine=True) for _ in range(n_mods)])
        self.bypass_conv1 = nn.ModuleList([nn.Conv3d(in_ch_each, in_ch_each, 1) for _ in range(n_mods)])
        self.bypass_conv2 = nn.ModuleList([nn.Conv3d(in_ch_each, gate_out_ch, 1) for _ in range(n_mods)])

        # 可选的后置对齐（保持 E 不变，预留接口）
        self.post_proj = nn.ModuleList([nn.Conv3d(embed_dim, embed_dim, 1) for _ in range(n_mods)])

    def forward(self, xs: List[torch.Tensor], target_shape: Optional[Tuple[int, int, int]] = None,
                mask: Optional[List[bool]] = None) -> torch.Tensor:
        """
        xs: [x0,x1,x2,x3]，每个 (B,1,D,H,W)
        target_shape: (Db,Hb,Wb)
        mask: 缺失模态标志（True 可用 / False 缺失）
        """
        if mask is None:
            mask = [True] * self.n
        device = xs[0].device
        B = xs[0].shape[0]

        # 1) 适配瓶颈尺度 + token 化 + 线性，记全局描述符 desc_i；旁路 U_i
        tokens, infos, descs, bypass_feats = [None]*self.n, [None]*self.n, [None]*self.n, [None]*self.n
        for i in range(self.n):
            if not mask[i]:
                continue
            xi = xs[i]                        # (B,1,D,H,W)
            if target_shape is not None:
                xi_b = F.adaptive_avg_pool3d(xi, target_shape)  # (B,1,Db,Hb,Wb)
            else:
                xi_b = xi

            tok, info = self.tokenizer.patchify(xi_b)           # (B,N,1*P)
            z = self.in_proj[i](tok)                            # (B,N,E)
            desc = z.mean(dim=1)                                # (B,E)
            tokens[i], infos[i], descs[i] = z, info, desc

            # 旁路 U_i：IN -> 1x1 -> 1x1  (输出 E 或 1 通道)
            u = self.bypass_norm[i](xi_b)
            u = self.bypass_conv1[i](u)
            u = self.bypass_conv2[i](u)                         # (B,E,Db,Hb,Wb) 或 (B,1,Db,Hb,Wb)
            bypass_feats[i] = u

        # 2) 条件化 Mamba（FiLM）
        Hs = [None]*self.n
        for i in range(self.n):
            if not mask[i]:
                continue
            Hs[i] = self.cond_mamba[i](tokens[i], descs[i])     # (B,N,E)

        # 3) α_ij by desc 相似度（masked softmax）
        S = torch.zeros((B, self.n, self.n), device=device)
        for i in range(self.n):
            if not mask[i]:
                continue
            for j in range(self.n):
                if not mask[j]:
                    continue
                S[:, i, j] = (descs[i] * descs[j]).sum(-1)      # 点积
        big_neg = -1e9
        Sm = S.clone()
        for i in range(self.n):
            for j in range(self.n):
                if not mask[j]:
                    Sm[:, i, j] = big_neg
        A = torch.softmax(Sm, dim=-1)                           # (B,n,n)

        # 4) 低秩 C_j + 跨模态读出： y_i = D_i x_i + Σ_j α_ij · (h_i @ C_j)
        Ys = [None]*self.n
        for i in range(self.n):
            if not mask[i]:
                continue
            Di_xi = self.D_lin[i](tokens[i])                    # (B,N,E)
            y_i = Di_xi
            for j in range(self.n):
                if not mask[j]:
                    continue
                Uj = self.readout_U[j](descs[j]).view(B, self.embed_dim, -1)  # (B,E,r)
                Vj = self.readout_V[j](descs[j]).view(B, self.embed_dim, -1)  # (B,E,r)
                Cj = torch.matmul(Uj, Vj.transpose(1, 2))                      # (B,E,E)
                proj = torch.matmul(Hs[i], Cj)                                 # (B,N,E)
                aij = A[:, i, j].view(B, 1, 1)                                 # (B,1,1)
                y_i = y_i + aij * proj
            Ys[i] = y_i

        # 5) unpatchify -> Y_i；6) 旁路重要性门控：Ẏ_i = IN(Y_i)，M_i = SiLU(U_i)，Z_i = M_i ⊙ Ẏ_i
        fused = 0.0
        for i in range(self.n):
            if not mask[i]:
                continue
            vol = PatchTokenizer3D.unpatchify(Ys[i], infos[i], out_channels=self.embed_dim)  # (B,E,Db_pad,Hb_pad,Wb_pad)
            vol = PatchTokenizer3D.crop_to_orig(vol, infos[i]["orig"])                        # (B,E,Db,Hb,Wb)
            vol = self.out_norm(vol)                                                          
            M = torch.nn.functional.silu(bypass_feats[i])                                     # (B,E,Db,Hb,Wb) 或 (B,1,Db,Hb,Wb)
            if M.shape[1] == 1:
                M = M.expand_as(vol)
            vol = M * vol                                                                     # Z_i
            vol = self.post_proj[i](vol)                                                      
            fused = fused + vol

        return fused  # (B,E,Db,Hb,Wb)


# ------------------------------
# 编码器：每层对每模态做 Down->GSC->DirectionalMamba3D->FRM；
# 融合策略：4模态 concat -> 1x1x1 -> GroupNorm -> C
# ------------------------------
class MambaEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, depths: List[int] = [1, 1, 1, 1],
                 dims: List[int] = [48, 96, 192, 384], stage_fuse: str = "concat+1x1",
                 drop_path: float = 0.0):
        super().__init__()
        self.stage_fuse = stage_fuse
        self.dims = dims

        # 4 层下采样（共享参数，4 模态共用）
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # 每层模块：GSC、LMB(C->C)、FRM(C->C)、融合降维
        self.gscs = nn.ModuleList()
        self.lmbs = nn.ModuleList()
        self.frms = nn.ModuleList()
        self.post_reduce = nn.ModuleList()  # 融合后降维到 dims[i]
        self.carry_reduce = nn.ModuleList() # 给下一层的 1×1 可学习通道对齐

        for i in range(4):
            C = dims[i]
            self.gscs.append(GSC(C, drop_path=drop_path))
            self.lmbs.append(
                DirectionalMamba3D(
                    dim=C, d_state=16, d_conv=4, expand=2,
                    pool_reduce=1, res_init=1e-3, drop_path=drop_path
                )
            )
            self.frms.append(FRM(C, use_instance_norm=True, dw_k=5, res_init=1e-3, drop_path=drop_path))
            if self.stage_fuse == "concat+1x1":
                self.post_reduce.append(
                    nn.Sequential(
                        nn.Conv3d(4 * C, C, kernel_size=1),
                        nn.GroupNorm(num_groups=min(8, C), num_channels=C),
                    )
                )
            else:
                self.post_reduce.append(nn.Identity())
            self.carry_reduce.append(nn.Conv3d(C, C, 1))

        # 与 UNETR 跳连接一致
        self.norms = nn.ModuleList([nn.InstanceNorm3d(dims[i]) for i in range(4)])
        self.mlps = nn.ModuleList([MlpChannel(dims[i], 2 * dims[i]) for i in range(4)])

    def _run_one_modality_one_stage(self, x: torch.Tensor, i_stage: int) -> torch.Tensor:
        x = self.downsample_layers[i_stage](x)  # /2, /4, /8, /16
        x = self.gscs[i_stage](x)
        x = self.lmbs[i_stage](x)               # (B, C, D,H,W)
        x = self.frms[i_stage](x)               # (B, C, D,H,W)
        return x

    def forward_features(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i in range(4):
            f0 = self._run_one_modality_one_stage(x0, i)
            f1 = self._run_one_modality_one_stage(x1, i)
            f2 = self._run_one_modality_one_stage(x2, i)
            f3 = self._run_one_modality_one_stage(x3, i)

            if self.stage_fuse == "concat+1x1":
                x = torch.cat([f0, f1, f2, f3], dim=1)   # (B, 4C, ...)
                x = self.post_reduce[i](x)               # (B, C, ...)
            else:
                x = f0                                   # (B, C, ...)

            x = self.norms[i](x)
            x = self.mlps[i](x)
            outs.append(x)

            # 下一层输入：用 1×1 可学习压缩对齐
            x0 = self.carry_reduce[i](f0)
            x1 = self.carry_reduce[i](f1)
            x2 = self.carry_reduce[i](f2)
            x3 = self.carry_reduce[i](f3)
        return tuple(outs)

    def forward(self, x_in0: torch.Tensor, x_in1: torch.Tensor, x_in2: torch.Tensor, x_in3: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward_features(x_in0, x_in1, x_in2, x_in3)


# ------------------------------
# MprmMamba（整合稳态改造）
# ------------------------------
class MprmMamba(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 13,
        depths: List[int] = [1, 1, 1, 1],
        feat_size: List[int] = [48, 96, 192, 384],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 3,
        stage_fuse: str = "concat+1x1",
        use_mprm: bool = True,
        mprm_embed: int = 64,
        mprm_patch: Tuple[int, int, int] = (2, 2, 2),
        mprm_lowrank: int = 32,
        mprm_gate_mode: str = "channel",   # or "spatial"
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims

        # Encoder（每模态内部三方向 LMB，已稳态化）
        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            stage_fuse=stage_fuse,
            drop_path=drop_path_rate,
        )

        # UNETR 风格的浅层与跳连分支
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=4, out_channels=self.feat_size[0],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[1],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[2],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[3],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.hidden_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )

        # Decoder
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.feat_size[2],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[1],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[0],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[0],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

        # 瓶颈 MPRM
        self.use_mprm = use_mprm
        if self.use_mprm:
            self.mprm = MPRM_Perspective(
                in_ch_each=1, embed_dim=mprm_embed, n_mods=4,
                patch_size=mprm_patch, lowrank=mprm_lowrank,
                gate_mode=mprm_gate_mode,
            )
            self.mprm_gate = ResidualScale(0.0)                # zero-init gate
            self.mprm_proj = nn.Conv3d(mprm_embed, self.feat_size[3], kernel_size=1)
        else:
            self.mprm = None

    def forward(self, x_in0: torch.Tensor, x_in1: torch.Tensor, x_in2: torch.Tensor, x_in3: torch.Tensor,
                modality_mask: Optional[List[bool]] = None) -> torch.Tensor:
        """
        x_in{0..3}: (B,1,D,H,W), 四个模态输入（例如 T1, T1c, T2, FLAIR）
        modality_mask: 可选的缺失模态掩码（长度4，True可用/False缺失）
        """
        # 编码金字塔 (/2,/4,/8,/16)
        outs = self.vit(x_in0, x_in1, x_in2, x_in3)  # 每层输出 (B, C_i, ...)

        # 浅层 skip（四模态拼为4通道喂入）
        x_combined = torch.cat([x_in0, x_in1, x_in2, x_in3], dim=1)  # (B,4,D,H,W)
        enc1 = self.encoder1(x_combined)  # full-res skip

        x2 = outs[0]                      # /2
        enc2 = self.encoder2(x2)

        x3 = outs[1]                      # /4
        enc3 = self.encoder3(x3)

        x4 = outs[2]                      # /8
        enc4 = self.encoder4(x4)

        # 瓶颈 (/16)
        f3 = outs[3]
        if self.use_mprm:
            bott_shape = f3.shape[-3:]  # (Db,Hb,Wb)
            z_readout = self.mprm([x_in0, x_in1, x_in2, x_in3],
                                  target_shape=bott_shape,
                                  mask=modality_mask)            # (B, E, Db, Hb, Wb)
            z_readout = self.mprm_proj(z_readout)                # 通道对齐到 feat_size[3]
            f3 = f3 + self.mprm_gate(z_readout)                  # 门控残差融合

        enc_hidden = self.encoder5(f3)

        # 解码
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        return self.out(out)
