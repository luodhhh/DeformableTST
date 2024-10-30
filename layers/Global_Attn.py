import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from layers.Transformer_Module import LayerNormProxy

class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()


        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads

        self.q_size = q_size

        self.kv_size = self.q_size // stride
        self.nc = n_head_channels * n_heads



        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.log_cpb = log_cpb
        self.dwc_pe = dwc_pe


        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups


        self.stride = stride
        self.ksize = ksize
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0


        self.conv_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv1d(self.n_group_channels, 1, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)


        self.proj_q = nn.Conv1d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv1d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv1d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv1d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )


        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)


        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv1d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_size, self.kv_size)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                self.rpe_table = nn.Sequential(
                    nn.Linear(1, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_size * 2 - 1, 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, L_key, B, dtype, device):

        ref = torch.linspace(0.5, L_key - 0.5, L_key, dtype=dtype, device=device)
        ref.div_(L_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, :, None, None].expand(B * self.n_groups, -1, -1, -1)

        return ref

    @torch.no_grad()
    def _get_q_grid(self, L, B, dtype, device):

        ref = torch.arange(0, L, dtype=dtype, device=device)


        ref.div_(L - 1.0).mul_(2.0).sub_(1.0)

        ref = ref[None, :, None, None].expand(B * self.n_groups, -1, -1, -1)

        return ref

    def forward(self, x):


        B, C, L = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) n -> (b g) c n', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()
        Lk = offset.size(2)
        n_sample = Lk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Lk - 1.0)], device=device).reshape(1, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p n -> b n p')
        reference = self._get_ref_points(Lk, B, dtype, device)
        offset = offset.unsqueeze(-2)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Lk, f"Size is {x_sampled.size()}"
        else:
            pos_2 = pos.repeat(1,1,1,2)
            pos_2[... , 1] = 0
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, L, 1),
                grid=pos_2,
                mode='bilinear', align_corners=True)

        x_sampled = x_sampled.squeeze(-1)
        x_sampled = x_sampled.reshape(B, C, n_sample)


        q = q.reshape(B * self.n_heads, self.n_head_channels, L)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, L)).reshape(B * self.n_heads, self.n_head_channels, L)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, L, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(L, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, L, 1).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 1).unsqueeze(1)).mul(4.0)
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(L, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, L, 1).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 1).unsqueeze(1)).mul(0.5)

                displacement_2 = displacement.repeat(1, 1, 1, 2)
                displacement_2[..., 1] = 0
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
                    grid=displacement_2,
                    mode='bilinear', align_corners=True)

                attn_bias = attn_bias.reshape(B * self.n_heads, L, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)


        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, L)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Lk, 1), reference.reshape(B, self.n_groups, Lk, 1)