# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# The helper classes below are adapted from the hi-ml-multimodal project:
# https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
# We vendor a minimal subset (ResNet encoder + temporal ViT pooler) so that
# BioViL-T image checkpoints can be loaded without pulling the entire package.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetHIML(ResNet):
    """Torchvision ResNet with access to intermediate feature maps."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        return_intermediate_layers: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_intermediate_layers:
            return x0, x1, x2, x3, x4
        return x4


def build_resnet50_backbone() -> ResNetHIML:
    return ResNetHIML(block=Bottleneck, layers=[3, 4, 6, 3])


@dataclass
class _AttentionOutput:
    output: torch.Tensor
    attention: Optional[torch.Tensor] = None


class VisionTransformerPooler(nn.Module):
    """Temporal ViT that fuses current/previous frame features into patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        grid_shape: Tuple[int, int],
        num_heads: int = 8,
        num_blocks: int = 3,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        norm_factory = partial(nn.LayerNorm, eps=1e-6)
        block_kwargs = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=1.0,
            drop=0.10,
            attn_drop=0.10,
            drop_path=0.25,
            act_layer=nn.GELU,
            norm_layer=norm_factory,
        )
        self.blocks = nn.ModuleList([_TransformerBlock(**block_kwargs) for _ in range(num_blocks)])
        self.norm_post = norm_factory(embed_dim)
        self.grid_shape = grid_shape
        self.num_patches = grid_shape[0] * grid_shape[1]

        num_series = 2
        self.type_embed = nn.Parameter(torch.zeros(num_series, 1, embed_dim))
        trunc_normal_(self.type_embed, std=0.02)

        sine_embed = _SinePositionEmbedding(embedding_dim=embed_dim // 2, normalize=True)
        pos_embed = sine_embed(mask=torch.ones([1, grid_shape[0], grid_shape[1]]))
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.pos_drop = nn.Dropout(p=0.10)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, current: torch.Tensor, previous: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, channels, height, width = current.shape
        assert (height, width) == self.grid_shape, (
            f"Expected patch grid {self.grid_shape}, got {(height, width)}"
        )

        current_tokens = current.view(batch, channels, -1).transpose(1, 2)
        pos_embed = self.pos_embed.repeat(batch, 1, 1)  # type: ignore
        tokens = current_tokens

        if previous is not None:
            assert previous.shape == current.shape, "Previous image must match current image shape"
            prev_tokens = previous.view(batch, channels, -1).transpose(1, 2)
            tokens = torch.cat([tokens, prev_tokens], dim=1)
            pos_embed = torch.cat([pos_embed, pos_embed], dim=1)

        type_embeddings = self._build_type_embeddings(tokens, previous is not None)
        x = self.pos_drop(tokens)

        for block in self.blocks:
            x = block(x, pos_embed + type_embeddings)

        x = self.norm_post(x)
        current_slice = slice(0, self.num_patches)
        current_tokens = x[:, current_slice]
        return current_tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def _build_type_embeddings(self, tokens: torch.Tensor, has_previous: bool) -> torch.Tensor:
        batch, length, _ = tokens.shape
        type_cur = self.type_embed[0].expand(batch, length, -1)
        if not has_previous:
            return type_cur
        type_prev = self.type_embed[1].expand(batch, length, -1)
        return torch.cat([type_cur, type_prev], dim=1)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
        act_layer: type[nn.Module],
        norm_layer: type[nn.LayerNorm],
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _AttentionLayer(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        if pos_embed is not None:
            x = x + pos_embed
        attn_out = self.attn.forward_as_mhsa(x).output
        x = residual + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _AttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> _AttentionOutput:
        batch, length, dim = q.shape

        def _reshape(x: torch.Tensor) -> torch.Tensor:
            return (
                x.reshape(batch, length, self.num_heads, dim // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        q_proj = _reshape(self.proj_q(q))
        k_proj = _reshape(self.proj_k(k))
        v_proj = _reshape(self.proj_v(v))

        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v_proj).transpose(1, 2).reshape(batch, length, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return _AttentionOutput(output=out, attention=attn)

    def forward_as_mhsa(self, x: torch.Tensor) -> _AttentionOutput:
        return self(q=x, k=x, v=x)


class _SinePositionEmbedding:
    """2D sine-cosine positional embedding replicated from hi-ml."""

    def __init__(
        self,
        embedding_dim: int = 64,
        temperature: int = 10_000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * torch.pi

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        assert mask is not None, "No pixel mask provided"
        batch, height, width = mask.shape

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).view(batch, height * width, self.embedding_dim * 2)
        return pos


class BioViLTImageModel(nn.Module):
    """BioViL-T image encoder backed by a ResNet50 + temporal ViT pooler."""

    feature_dim: int = 512

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.encoder = _MultiImageEncoder()
        self.projector = _ProjectionMLP(
            input_dim=self.encoder.output_dim,
            hidden_dim=self.joint_dim,
            output_dim=self.joint_dim,
        )
        self._load_weights(checkpoint_path)

    @property
    def joint_dim(self) -> int:
        return 128

    def _load_weights(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_embeddings, pooled = self.encoder(images, return_patch_embeddings=True)
        _ = self.projector(patch_embeddings)  # projector kept for state dict compatibility
        return pooled


class _MultiImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = build_resnet50_backbone()
        self.backbone_to_vit = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.vit_pooler = VisionTransformerPooler(embed_dim=256, grid_shape=(14, 14))
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, 256, 1, 1))
        trunc_normal_(self.missing_previous_emb, std=0.02)
        self.output_dim = 512

    def forward(
        self,
        current_image: torch.Tensor,
        *,
        previous_image: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = current_image.shape[0]
        if previous_image is not None:
            assert current_image.shape == previous_image.shape, "Previous image must match current image shape"
            stacked = torch.cat([current_image, previous_image], dim=0)
            patch_stack = self.encoder(stacked)
            patch_stack = self.backbone_to_vit(patch_stack)
            patch_current, patch_prev = patch_stack[:batch], patch_stack[batch:]
            diff_features = self.vit_pooler(patch_current, patch_prev)
            patch_features = patch_current
        else:
            patch_features = self.encoder(current_image)
            patch_features = self.backbone_to_vit(patch_features)
            _, _, height, width = patch_features.shape
            diff_features = self.missing_previous_emb.repeat(batch, 1, height, width)

        fused = torch.cat([patch_features, diff_features], dim=1)
        pooled = torch.flatten(F.adaptive_avg_pool2d(fused, (1, 1)), 1)

        if return_patch_embeddings:
            return fused, pooled
        return fused, pooled


class _ProjectionMLP(nn.Module):
    """Minimal replica of hi-ml's convolutional MLP projector."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
