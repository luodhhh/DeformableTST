import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


from layers.RevIN import RevIN


from layers.Global_Attn import DAttentionBaseline
from layers.Transformer_Module import TransformerMLPWithConv, TransformerMLP, LayerNormProxy
from layers.head import Flatten_Head


from timm.models.layers import DropPath

class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1))
        else:
            return x * self.weight.view(-1, 1)


class Stage(nn.Module):

    def __init__(self,
                 fmap_size,
                 dim_embed, depths,
                 drop_path_rate, layer_scale_value,
                 use_pe,
                 use_lpu,local_kernel_size,
                 expansion, drop, use_dwc_mlp,
                 heads, attn_drop, proj_drop,
                 stage_spec,
                 window_size,
                 nat_ksize,
                 ksize, stride,
                 n_groups, offset_range_factor, no_off,
                 dwc_pe, fixed_pe, log_cpb,
                 ):


        super(Stage,self).__init__()
        fmap_size = fmap_size
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu


        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for d in range(2 * depths)]
        )


        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv1d(dim_embed, dim_embed, kernel_size=local_kernel_size, stride=1, padding=local_kernel_size//2,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )


        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()

        for i in range(depths):

            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')


            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())


        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop, local_kernel_size) for _ in range(depths)
            ]
        )

        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )

    def forward(self, x):


        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'D':

                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0

                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class DeformableTST(nn.Module):
    def __init__(self,
                 n_vars, rev, revin_affine, revin_subtract_last,
                 stem_ratio,
                 down_ratio,
                 fmap_size,
                 dims, depths,
                 drop_path_rate, layer_scale_value,
                 use_pe,
                 use_lpu,
                 local_kernel_size,
                 expansion, drop, use_dwc_mlp,
                 heads, attn_drop, proj_drop,
                 stage_spec,
                 window_size,
                 nat_ksize,
                 ksize, stride,
                 n_groups, offset_range_factor, no_off,
                 dwc_pe, fixed_pe, log_cpb,
                 seq_len, pred_len, head_dropout,
                 head_type, use_head_norm,
                 ):

        super(DeformableTST, self).__init__()

        self.rev = rev
        if self.rev:
            self.revin = RevIN(n_vars, affine=revin_affine, subtract_last=revin_subtract_last)


        patch_stride = stem_ratio
        patch_size = stem_ratio
        downsample_ratio = down_ratio
        self.downsample_layers = nn.ModuleList()
        if stem_ratio > 1:
            stem = nn.Sequential(
                nn.Conv1d(1, dims[0]//2, kernel_size = 3, stride = 2, padding= 3//2),
                LayerNormProxy(dims[0]//2),
                nn.GELU(),
                nn.Conv1d(dims[0]//2, dims[0], kernel_size=patch_size // 2, stride=patch_stride // 2),
                LayerNormProxy(dims[0])
            )
        else:
            stem = nn.Sequential(
                nn.Conv1d(1, dims[0], kernel_size = 1, stride = 1),
                LayerNormProxy(dims[0]))

        self.downsample_layers.append(stem)

        self.num_stage = len(depths)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                    LayerNormProxy(dims[i+1]),
                )
                self.downsample_layers.append(downsample_layer)



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        fmap_size = fmap_size // stem_ratio

        self.stages = nn.ModuleList()
        for i in range(self.num_stage):
            self.stages.append(
                Stage(fmap_size,
                     dims[i], depths[i],
                     dpr[sum(depths[:i]):sum(depths[:i + 1])],
                     layer_scale_value[i],
                     use_pe[i],

                     use_lpu[i],
                     local_kernel_size[i],

                     expansion, drop, use_dwc_mlp[i],

                     heads[i], attn_drop, proj_drop,

                     stage_spec[i],
                     window_size[i],
                     nat_ksize[i],

                     ksize[i], stride[i],
                     n_groups[i], offset_range_factor[i], no_off[i],
                     dwc_pe[i], fixed_pe[i], log_cpb[i],
                                )
            )
            fmap_size = fmap_size // down_ratio

        self.use_head_norm = use_head_norm
        if use_head_norm:
            self.head_norm = LayerNormProxy(dims[self.num_stage -1])
        if head_type == 'Flatten':
            L = seq_len
            S = 1
            for i in range(self.num_stage):
                if i == 0:
                    S = S * stem_ratio
                else:
                    S = S *down_ratio

            if L % S == 0:
                N = L // S
            else:
                N = L // S + 1
            self.head = Flatten_Head(individual=False, n_vars=n_vars, nf=N*dims[self.num_stage -1], target_window= pred_len, head_dropout= head_dropout)

        self.head_type = head_type

    def forward(self,x):

        if self.rev:
            x = self.revin(x,'norm')

        B,L,M = x.shape
        x = x.permute(0,2,1).reshape(B*M,L).unsqueeze(1)
        x = self.downsample_layers[0](x)


        for i in range(self.num_stage):
            x = self.stages[i](x)
            if i < (self.num_stage -1):
                x = self.downsample_layers[i+1](x)

        if self.head_type == 'Flatten':
            _,D,N = x.shape
            if self.use_head_norm:
                x = self.head_norm(x)
            x = x.reshape(B,M,D,N)
            x = self.head(x)
            x = x.permute(0,2,1)

        if self.rev:
            x = self.revin(x,'denorm')

        return x

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()

        n_vars = configs.n_vars
        rev = configs.revin
        revin_affine = configs.revin_affine
        revin_subtract_last = configs.revin_subtract_last

        stem_ratio = configs.stem_ratio
        down_ratio = configs.down_ratio

        fmap_size = configs.fmap_size

        dims = configs.dims
        depths = configs.depths

        drop_path_rate = configs.drop_path_rate
        layer_scale_value = configs.layer_scale_value

        use_pe = configs.use_pe

        use_lpu = configs.use_lpu
        local_kernel_size = configs.local_kernel_size

        expansion = configs.expansion
        drop = configs.drop
        use_dwc_mlp = configs.use_dwc_mlp

        heads = configs.heads
        attn_drop = configs.attn_drop
        proj_drop = configs.proj_drop

        stage_spec = configs.stage_spec

        window_size = configs.window_size
        nat_ksize = configs.nat_ksize

        ksize = configs.ksize
        stride = configs.stride
        n_groups = configs.n_groups
        offset_range_factor = configs.offset_range_factor
        no_off = configs.no_off

        dwc_pe = configs.dwc_pe
        fixed_pe = configs.fixed_pe
        log_cpb = configs.log_cpb

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        head_dropout = configs.head_dropout
        head_type = configs.head_type
        use_head_norm = configs.use_head_norm


        self.model = DeformableTST(
            n_vars, rev, revin_affine, revin_subtract_last,
            stem_ratio,
            down_ratio,
            fmap_size,
            dims, depths,
            drop_path_rate, layer_scale_value,
            use_pe,
            use_lpu,
            local_kernel_size,
            expansion, drop, use_dwc_mlp,
            heads, attn_drop, proj_drop,
            stage_spec,
            window_size,
            nat_ksize,
            ksize, stride,
            n_groups, offset_range_factor, no_off,
            dwc_pe, fixed_pe, log_cpb,
            seq_len, pred_len, head_dropout,
            head_type, use_head_norm,
        )

    def forward(self,x):

        return self.model(x)

