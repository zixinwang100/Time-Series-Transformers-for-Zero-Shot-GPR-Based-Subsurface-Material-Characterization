import torch
import torch.nn as nn
from typing import Optional

from .PatchTST_backbone import PatchTST_backbone
from .PatchTST_layers import series_decomp

class PatchTSTMaterialRegressor(nn.Module):
    """
    PatchTST encoder -> global regression head.

    Requires you modified PatchTST_backbone.forward as:
        def forward(self, z, return_feat=False):
            ...
            z = self.backbone(z)  # [B,C,d_model,patch_num]
            if return_feat: return z
            z = self.head(z)
            ...
            return z

    Inputs:
      x: [B,L] or [B,L,C] or [B,C,L]
    Output:
      y: [B,out_dim]
    """
    def __init__(
        self,
        seq_len: int,
        out_dim: int,
        c_in: int = 1,
        # PatchTST backbone params
        patch_len: int = 32,
        stride: int = 8,
        max_seq_len = None,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch=None,  # None or 'end'
        # RevIN
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        # Optional decomposition
        decomposition: bool = False,
        kernel_size: int = 25,
        # Pooling across channels (vars)
        pool_vars: str = "mean",   # "mean" or "concat"
        # Regression head params
        head_hidden: int = 256,
        head_drop: float = 0.1,
        # Extra kwargs forwarded to backbone (rarely needed)
        **kwargs,
    ):
        super().__init__()
        assert pool_vars in ["mean", "concat"], "pool_vars must be 'mean' or 'concat'"

        self.seq_len = seq_len
        self.c_in = c_in
        self.out_dim = out_dim
        self.pool_vars = pool_vars
        self.decomposition = decomposition

        if max_seq_len is None:
            max_seq_len = seq_len

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)

        # We still pass head_type='flatten' etc., but we will NOT use the forecasting head.
        # We only call forward(..., return_feat=True)
        self.backbone = PatchTST_backbone(
            c_in=c_in,
            context_window=seq_len,
            target_window=1,          # irrelevant for regression, but required by init
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=None,
            d_v=None,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=False,
            head_type="flatten",
            individual=False,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            verbose=False,
            **kwargs,
        )

        # Determine patch_num exactly like their backbone does
        patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == "end":
            patch_num += 1

        self._feat_dim_per_var = d_model * patch_num
        feat_dim = self._feat_dim_per_var if pool_vars == "mean" else self._feat_dim_per_var * c_in

        self.reg_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(head_hidden, out_dim),
            # nn.Sigmoid(),
            # nn.Softplus(),
        )

    @staticmethod
    def _to_bcl(x: torch.Tensor, seq_len: int, c_in: int) -> torch.Tensor:
        """
        Convert x to [B,C,L].
        Accept:
          - [B,L]
          - [B,L,C]
          - [B,C,L]
        """
        if x.ndim == 2:
            # [B,L] -> [B,1,L]
            return x.unsqueeze(1)

        if x.ndim == 3:
            B = x.shape[0]
            # Heuristic: if middle dim equals seq_len, assume [B,L,C]
            if x.shape[1] == seq_len and x.shape[2] == c_in:
                return x.permute(0, 2, 1)  # [B,C,L]
            # else assume already [B,C,L]
            return x

        raise ValueError(f"Unexpected input shape {tuple(x.shape)}; expected [B,L], [B,L,C], or [B,C,L].")

    def _encode_feat(self, x_bcl: torch.Tensor) -> torch.Tensor:
        """
        x_bcl: [B,C,L]
        return: feat [B,C,d_model,patch_num]
        """
        feat = self.backbone(x_bcl, return_feat=True)
        if feat.ndim != 4:
            raise RuntimeError(f"Expected feat [B,C,d_model,patch_num], got {tuple(feat.shape)}")
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to [B,C,L]
        x_bcl = self._to_bcl(x, self.seq_len, self.c_in)

        if self.decomposition:
            # series_decomp expects [B,L,C]
            x_blc = x_bcl.permute(0, 2, 1)
            res, trend = self.decomp_module(x_blc)      # [B,L,C] each
            res_bcl = res.permute(0, 2, 1)
            trend_bcl = trend.permute(0, 2, 1)
            feat = self._encode_feat(res_bcl) + self._encode_feat(trend_bcl)
        else:
            feat = self._encode_feat(x_bcl)

        # feat: [B,C,d_model,patch_num] -> flatten over (d_model,patch_num)
        B, C, D, P = feat.shape
        feat = feat.reshape(B, C, D * P)               # [B,C,F]

        # Pool across variables/channels
        if self.pool_vars == "mean":
            feat = feat.mean(dim=1)                    # [B,F]
        else:
            feat = feat.reshape(B, -1)                 # [B,C*F]

        y = self.reg_head(feat)                        # [B,out_dim]
        return y