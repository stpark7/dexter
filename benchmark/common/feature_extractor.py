import torch
import torch.nn as nn
import torch.nn.functional as F
from point_e.models.download import load_checkpoint

from .pointnet2_utils import PointNetSetAbstraction


class PointNetClassifier(nn.Module):
    def __init__(self, num_class, normal_channel=True, width_mult=1):
        super(PointNetClassifier, self).__init__()
        self.width_mult = width_mult
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64 * width_mult, 64 * width_mult, 128 * width_mult],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 * width_mult + 3,
            mlp=[128 * width_mult, 128 * width_mult, 256 * width_mult],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 * width_mult + 3,
            mlp=[256 * width_mult, 512 * width_mult, 1024 * width_mult],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024 * width_mult, 512 * width_mult)
        self.bn1 = nn.BatchNorm1d(512 * width_mult)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512 * width_mult, 256 * width_mult)
        self.bn2 = nn.BatchNorm1d(256 * width_mult)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256 * width_mult, num_class)

    def forward(self, xyz, features=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024 * self.width_mult)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        result_features = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(result_features))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        if features:
            return x, l3_points, result_features
        else:
            return x, l3_points


def get_model(device="cuda", cache_dir=None):
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the numpy
    # objects stored in the point_e pointnet checkpoint. load_checkpoint() doesn't expose
    # the weights_only kwarg, so temporarily force it off for this trusted checkpoint.
    orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _load
    try:
        ckpt = load_checkpoint("pointnet", device=torch.device("cpu"), cache_dir=cache_dir)
    finally:
        torch.load = orig_load
    state_dict = ckpt["model_state_dict"]
    device = device
    model = PointNetClassifier(num_class=40, normal_channel=False, width_mult=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def normalize_point_clouds(pc: torch.Tensor) -> torch.Tensor:
    """Center each point cloud and scale to unit max radius (FID feature prep)."""
    centroids = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroids
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / m
    return pc
