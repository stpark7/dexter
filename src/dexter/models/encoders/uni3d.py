import os
from types import SimpleNamespace

import numpy as np
import timm
import torch
import torch.nn as nn
import torch_fpsample
from huggingface_hub import hf_hub_download

from dexter.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def fps(data, number):
    """
    Farthest Point Sampling using the new FarthestPointSampling function.

    Args:
        data: Input point cloud tensor of shape (B, N, 3)
        number: Number of points to sample

    Returns:
        fps_data: Sampled points tensor of shape (B, number, 3)
    """
    device = data.device
    sampled_points, indices = torch_fpsample.sample(data.cpu(), number)
    return sampled_points.to(device)


# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
    return dist


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        log.info(f"patch dropout prob is {prob}")

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, config):
        super().__init__()

        self.trans_dim = config.pc_feat_dim  # 768
        self.embed_dim = config.embed_dim  # 512
        self.group_size = config.group_size  # 32
        self.num_group = config.num_group  # 512
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dim = config.pc_encoder_dim  # 256
        self.encoder = Encoder(encoder_channel=self.encoder_dim)

        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)

        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = (
            PatchDropout(config.patch_dropout) if config.patch_dropout > 0.0 else nn.Identity()
        )
        self.visual = point_transformer

    def forward(self, pts, colors, return_patch_embeddings=False):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)

        # ModuleList not support forward
        for _, blk in enumerate(self.visual.blocks):
            x = blk(x)

        if return_patch_embeddings:
            # Return all patch embeddings (excluding CLS token)
            x = self.visual.norm(x[:, 1:, :])  # Skip CLS token at index 0
            x = self.visual.fc_norm(x)
        else:
            # Return CLS token embedding only
            x = self.visual.norm(x[:, 0, :])
            x = self.visual.fc_norm(x)

        x = self.trans2embed(x)
        return x


class Uni3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        config = get_config(cfg.variant)

        point_transformer = timm.create_model(
            config.pc_model,
            checkpoint_path=getattr(config, "pretrained_pc", None),
            drop_path_rate=getattr(config, "drop_path_rate", 0.0),
        )

        # create whole point cloud encoder
        self.point_encoder = PointcloudEncoder(point_transformer, config)

    @property
    def point_feat_dim(self):
        return self.point_encoder.embed_dim

    def encode_pc(self, pc, **kwargs):
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        pc_feat = self.point_encoder(xyz, color, return_patch_embeddings=True)
        return pc_feat

    def forward(self, pc, *args, **kwargs):
        return self.encode_pc(pc)

    @classmethod
    def from_pretrained(cls, cfg, checkpoint_path=None, cache_dir=None, force_download=False):
        """
        Load a pretrained Uni3D model from Hugging Face Hub.

        Args:
            cfg: Configuration object with fields:
                - variant: Model variant ("tiny", "small", "base", "large")
                - Optional: checkpoint_path, cache_dir, force_download
            checkpoint_path (str, optional): Path to local checkpoint file.
                If provided, will use this instead of downloading.
            cache_dir (str, optional): Directory to cache downloads.
                Defaults to ~/.cache/huggingface/hub
            force_download (bool): Force re-download even if file exists

        Returns:
            Uni3D: Initialized model with pretrained weights

        Example:
            >>> model = Uni3D.from_pretrained(config.encoder)
        """
        # Extract parameters from config
        variant = getattr(cfg, "variant", "base")
        checkpoint_path = getattr(cfg, "checkpoint_path", checkpoint_path)
        cache_dir = getattr(cfg, "cache_dir", cache_dir)
        force_download = getattr(cfg, "force_download", force_download)

        # Map variant to full model variant name for checkpoint download
        variant_to_model = {
            "tiny": "uni3d-ti",
            "small": "uni3d-s",
            "base": "uni3d-b",
            "large": "uni3d-l",
        }
        model_variant = variant_to_model.get(variant, "uni3d-b")

        # Create the model with the provided config
        log.info(f"Creating Uni3D model with config variant: {variant}")
        model = cls(cfg)

        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = download_uni3d_checkpoint(
                model_variant=model_variant,
                cache_dir=cache_dir,
                force_download=force_download,
            )

        # Load checkpoint weights
        load_uni3d_checkpoint(model, checkpoint_path, strict=False)

        return model


def download_uni3d_checkpoint(model_variant="uni3d-b", cache_dir=None, force_download=False):
    """
    Download Uni3D model checkpoint from Hugging Face.

    Args:
        model_variant (str): Model variant to download. Options:
            - "uni3d-ti": Tiny model
            - "uni3d-s": Small model
            - "uni3d-b": Base model (default)
            - "uni3d-l": Large model
            - Add "-no-lvis" suffix for versions without LVIS training
        cache_dir (str, optional): Directory to cache downloads.
            Defaults to ~/.cache/huggingface/hub
        force_download (bool): Force re-download even if file exists

    Returns:
        str: Path to downloaded checkpoint file
    """
    repo_id = "BAAI/Uni3D"
    filename = f"modelzoo/{model_variant}/model.pt"

    log.info(f"Downloading Uni3D {model_variant} checkpoint from {repo_id}")

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )

        log.info(f"Successfully downloaded checkpoint to: {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        log.error(f"Failed to download checkpoint: {e}")
        raise RuntimeError(
            f"Could not download Uni3D checkpoint {model_variant} from {repo_id}"
        ) from e


def load_uni3d_checkpoint(model, checkpoint_path, strict=True):
    """
    Load Uni3D checkpoint into model.

    Args:
        model: Uni3D model instance
        checkpoint_path (str): Path to checkpoint file
        strict (bool): Whether to strictly enforce state dict keys match

    Returns:
        dict: Loaded checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading Uni3D checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["module"], strict=strict)

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        log.info("Successfully loaded Uni3D checkpoint")

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "checkpoint_info": checkpoint.get("info", {}) if isinstance(checkpoint, dict) else {},
        }

    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}") from e


def get_config(model_type="base"):
    """
    Get configuration arguments for Uni3D model by model type.

    Args:
        model_type (str): Model type. Options:
            - "tiny": Tiny model (eva02_tiny_patch14_224, 192 dims)
            - "small": Small model (eva02_small_patch14_224, 384 dims)
            - "base": Base model (eva02_base_patch14_448, 768 dims)
            - "large": Large model (eva02_large_patch14_448, 1024 dims)
            - "giant": Giant model (eva_giant_patch14_560, 1408 dims)

    Returns:
        SimpleNamespace: Configuration object with model parameters
    """
    model_configs = {
        "tiny": {
            "pc_model": "eva02_tiny_patch14_224",
            "pc_feat_dim": 192,
            "model_variant": "uni3d-ti",
        },
        "small": {
            "pc_model": "eva02_small_patch14_224",
            "pc_feat_dim": 384,
            "model_variant": "uni3d-s",
        },
        "base": {
            "pc_model": "eva02_base_patch14_448",
            "pc_feat_dim": 768,
            "model_variant": "uni3d-b",
        },
        "large": {
            "pc_model": "eva02_large_patch14_448",
            "pc_feat_dim": 1024,
            "model_variant": "uni3d-l",
        },
        "giant": {
            "pc_model": "eva_giant_patch14_560",
            "pc_feat_dim": 1408,
            "model_variant": "uni3d-l",  # Using large variant for giant
        },
    }

    if model_type not in model_configs:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {list(model_configs.keys())}"
        )

    config = model_configs[model_type]

    # Default parameters that can be overridden
    default_config = {
        "embed_dim": 1024,
        "group_size": 64,
        "num_group": 512,
        "pc_encoder_dim": 512,
        "patch_dropout": 0.0,
        "drop_path_rate": 0.0,
        "pretrained_pc": None,
    }

    # Merge with model-specific config
    default_config.update(config)

    return SimpleNamespace(**default_config)


def create_uni3d(config, pretrained=True, checkpoint_path=None):
    """
    Create Uni3D model with optional pretrained weights.

    Args:
        config: Configuration object with model parameters
        pretrained (bool): Whether to load pretrained weights
        checkpoint_path (str, optional): Path to specific checkpoint.
            If None and pretrained=True, will download default checkpoint.

    Returns:
        Uni3D: Initialized model
    """
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(
        config.pc_model,
        checkpoint_path=getattr(config, "pretrained_pc", None),
        drop_path_rate=getattr(config, "drop_path_rate", 0.0),
    )

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, config)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder)

    # Load pretrained weights if requested
    if pretrained:
        if checkpoint_path is None:
            # Download default checkpoint
            model_variant = getattr(config, "model_variant", "uni3d-b")
            checkpoint_path = download_uni3d_checkpoint(model_variant)

        load_uni3d_checkpoint(model, checkpoint_path, strict=False)

    return model
