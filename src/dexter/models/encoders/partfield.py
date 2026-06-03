import os
import time

import numpy as np
import torch
import torch.nn as nn
import trimesh
from huggingface_hub import hf_hub_download
from plyfile import PlyData, PlyElement

from dexter.utils.logger import RankedLogger

from .partfield_modules.model_utils import VanillaMLP
from .partfield_modules.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
from .partfield_modules.triplane import TriplaneTransformer, get_grid_coord

log = RankedLogger(__name__, rank_zero_only=True)


def get_config(*args, **kwargs):
    from easydict import EasyDict

    _c = {
        "seed": 0,
        "output_dir": "results",
        "result_name": "test_all",
        "triplet_sampling": "random",
        "load_original_mesh": False,
        "num_pos": 64,
        "num_neg_random": 256,
        "num_neg_hard_pc": 128,
        "num_neg_hard_emb": 128,
        "vertex_feature": False,
        "n_point_per_face": 1000,
        "n_sample_each": 10000,
        "preprocess_mesh": False,
        "regress_2d_feat": False,
        "is_pc": False,
        "cut_manifold": False,
        "remesh_demo": False,
        "correspondence_demo": False,
        "save_every_epoch": 10,
        "training_epochs": 30,
        "continue_training": False,
        "continue_ckpt": None,
        "epoch_selected": "epoch=50.ckpt",
        "triplane_resolution": 128,
        "triplane_channels_low": 128,
        "triplane_channels_high": 512,
        "lr": 1e-3,
        "train": True,
        "test": False,
        "inference_save_pred_sdf_to_mesh": True,
        "inference_save_feat_pca": True,
        "name": "test",
        "test_subset": False,
        "test_corres": False,
        "test_partobjaversetiny": False,
        "dataset": {
            "type": "Demo_Dataset",
            "data_path": "objaverse_data/",
            "train_num_workers": 64,
            "val_num_workers": 32,
            "train_batch_size": 2,
            "val_batch_size": 2,
            "all_files": [],
        },
        "voxel2triplane": {
            "transformer_dim": 1024,
            "transformer_layers": 6,
            "transformer_heads": 8,
            "triplane_low_res": 32,
            "triplane_high_res": 256,
            "triplane_dim": 64,
            "normalize_vox_feat": False,
        },
        "loss": {
            "triplet": 0.0,
            "sdf": 1.0,
            "feat": 10.0,
            "l1": 0.0,
        },
        "use_pvcnn": False,
        "use_pvcnnonly": True,
        "pvcnn": {
            "point_encoder_type": "pvcnn",
            "use_point_scatter": True,
            "z_triplane_channels": 256,
            "z_triplane_resolution": 128,
            "unet_cfg": {
                "depth": 3,
                "enabled": True,
                "rolled": True,
                "use_3d_aware": True,
                "start_hidden_channels": 32,
                "use_initial_conv": False,
            },
        },
        "use_2d_feat": False,
        "inference_metrics_only": False,
        "embed_dim": 1024,
        "model_variant": "base",
    }

    return EasyDict(_c)


class PartField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_config = get_config(cfg.variant)

        self.model_config = model_config
        self.automatic_optimization = False
        self.triplane_resolution = model_config.triplane_resolution
        self.triplane_channels_low = model_config.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=model_config.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=model_config.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(
            input_dim=64,
            output_dim=1,
            out_activation="tanh",
            n_neurons=64,  # 64
            n_hidden_layers=6,
        )  # 6
        self.use_pvcnn = model_config.use_pvcnnonly
        self.use_2d_feat = model_config.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                model_config.pvcnn,
                device="cuda",
                shape_min=-1,
                shape_length=2,
                use_2d_feat=self.use_2d_feat,
            )  # .cuda()
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction="none")

        if model_config.regress_2d_feat:
            self.feat_decoder = VanillaMLP(
                input_dim=64,
                output_dim=192,
                out_activation="GELU",
                n_neurons=64,  # 64
                n_hidden_layers=6,
            )  # 6

    @classmethod
    def from_pretrained(cls, cfg, checkpoint_path=None, cache_dir=None, force_download=False):
        """
        Load a pretrained PartField model.

        Args:
            cfg: Configuration object with fields like:
                - variant: Model variant (e.g., 'base')
                - downsample_patch_embeddings: Whether to downsample patch embeddings
                - normalize_point_cloud: Whether to normalize point cloud
            checkpoint_path: Optional path to checkpoint. If None, downloads from HuggingFace.
            cache_dir: Optional cache directory for downloaded checkpoints.
            force_download: If True, re-downloads checkpoint even if cached.

        Returns:
            PartField model with pretrained weights loaded.
        """
        # Create model instance with provided config
        model = cls(cfg)

        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = download_partfield_checkpoint(
                cache_dir=cache_dir, force_download=force_download
            )

        # Load checkpoint
        load_partfield_checkpoint(model, checkpoint_path, strict=True)

        log.info("Successfully created PartField model from pretrained checkpoint")

        return model

    @property
    def point_feat_dim(self):
        return 1024

    def encode_pc(self, pc, return_point_features=True):
        downsample_patch_embeddings = self.cfg.downsample_patch_embeddings
        normalize_point_cloud = self.cfg.normalize_point_cloud

        xyz = pc[:, :, :3].contiguous()
        if self.pvcnn.unet_encoder.conv_final.weight.dtype == torch.bfloat16:
            xyz = xyz.to(dtype=torch.bfloat16)

        pc_feat = self.pvcnn(xyz, xyz, normalize_point_cloud=normalize_point_cloud)
        planes = pc_feat
        planes, tokens = self.triplane_transformer(planes, return_tokens=True)

        # tokens: (B, 3072, D)
        if downsample_patch_embeddings:
            h = w = self.triplane_transformer.triplane_low_res
            b, d = tokens.shape[0], tokens.shape[-1]
            x = tokens.view(b, 3, h, w, -1)
            x = torch.einsum("nihwd->indhw", x)
            x = x.contiguous().view(3 * b, -1, h, w)
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            x = x.view(3, b, *x.shape[-3:])
            x = torch.einsum("indhw->nihwd", x)
            tokens = x.reshape(b, -1, d)

        if not return_point_features:
            return tokens

        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        point_feat = sample_triplane_feat(part_planes, xyz)  # N, M, C
        return tokens, point_feat

    def forward(self, pc, return_point_features=True):
        return self.encode_pc(pc, return_point_features=return_point_features)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        save_dir = f"exp_results/{self.cfg.result_name}"
        os.makedirs(save_dir, exist_ok=True)

        uid = batch["uid"][0]
        view_id = 0
        starttime = time.time()

        if uid in ["car", "complex_car"]:
            # if uid == "complex_car":
            print("Skipping this for now.")
            print(uid)
            return

        ### Skip if model already processed
        if os.path.exists(f"{save_dir}/part_feat_{uid}_{view_id}.npy") or os.path.exists(
            f"{save_dir}/part_feat_{uid}_{view_id}_batch.npy"
        ):
            print("Already processed " + uid)
            return

        N = batch["pc"].shape[0]  # noqa: N806
        assert N == 1

        if self.use_2d_feat:
            print("ERROR. Dataloader not implemented with input 2d feat.")
            exit()  # noqa: PLR1722
        else:
            pc_feat = self.pvcnn(batch["pc"], batch["pc"])

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        if self.cfg.is_pc:
            tensor_vertices = batch["pc"].reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat(part_planes, tensor_vertices)  # N, M, C
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

            np.save(f"{save_dir}/part_feat_{uid}_{view_id}.npy", point_feat)
            print(f"Exported part_feat_{uid}_{view_id}.npy")

            ###########
            from sklearn.decomposition import PCA

            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

            pca = PCA(n_components=3)

            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (
                data_reduced.max() - data_reduced.min()
            )
            colors_255 = (data_reduced * 255).astype(np.uint8)

            points = batch["pc"].squeeze().detach().cpu().numpy()

            if colors_255 is None:
                colors_255 = np.full_like(points, 255)  # Default to white color (255,255,255)
            else:
                assert colors_255.shape == points.shape, "Colors must have the same shape as points"

            # Convert to structured array for PLY format
            vertex_data = np.array(
                [(*point, *color) for point, color in zip(points, colors_255)],  # noqa: B905
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )

            # Create PLY element
            el = PlyElement.describe(vertex_data, "vertex")
            # Write to file
            filename = f"{save_dir}/feat_pca_{uid}_{view_id}.ply"
            PlyData([el], text=True).write(filename)
            print(f"Saved PLY file: {filename}")
            ############

        else:
            use_cuda_version = True
            if use_cuda_version:

                def sample_points(vertices, faces, n_point_per_face):
                    # Generate random barycentric coordinates
                    # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
                    n_f = faces.shape[0]
                    u = torch.sqrt(
                        torch.rand(
                            (n_f, n_point_per_face, 1),
                            device=vertices.device,
                            dtype=vertices.dtype,
                        )
                    )
                    v = torch.rand(
                        (n_f, n_point_per_face, 1),
                        device=vertices.device,
                        dtype=vertices.dtype,
                    )
                    w0 = 1 - u
                    w1 = u * (1 - v)
                    w2 = u * v

                    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
                    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
                    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
                    points = (
                        w0 * face_v_0.unsqueeze(dim=1)
                        + w1 * face_v_1.unsqueeze(dim=1)
                        + w2 * face_v_2.unsqueeze(dim=1)
                    )
                    return points

                def sample_and_mean_memory_save_version(
                    part_planes, tensor_vertices, n_point_per_face
                ):
                    n_sample_each = self.cfg.n_sample_each  # we iterate over this to avoid OOM
                    n_v = tensor_vertices.shape[1]
                    n_sample = n_v // n_sample_each + 1
                    all_sample = []
                    for i_sample in range(n_sample):
                        sampled_feature = sample_triplane_feat(
                            part_planes,
                            tensor_vertices[
                                :,
                                i_sample * n_sample_each : i_sample * n_sample_each + n_sample_each,
                            ],
                        )
                        assert sampled_feature.shape[1] % n_point_per_face == 0
                        sampled_feature = sampled_feature.reshape(
                            1, -1, n_point_per_face, sampled_feature.shape[-1]
                        )
                        sampled_feature = torch.mean(sampled_feature, axis=-2)
                        all_sample.append(sampled_feature)
                    return torch.cat(all_sample, dim=1)

                if self.cfg.vertex_feature:
                    tensor_vertices = batch["vertices"][0].reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(
                        part_planes, tensor_vertices, 1
                    )
                else:
                    n_point_per_face = self.cfg.n_point_per_face
                    tensor_vertices = sample_points(
                        batch["vertices"][0], batch["faces"][0], n_point_per_face
                    )
                    tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(
                        part_planes, tensor_vertices, n_point_per_face
                    )  # N, M, C

                #### Take mean feature in the triangle
                print("Time elapsed for feature prediction: " + str(time.time() - starttime))
                point_feat = point_feat.reshape(-1, 448).cpu().numpy()
                np.save(f"{save_dir}/part_feat_{uid}_{view_id}_batch.npy", point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                from sklearn.decomposition import PCA

                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (
                    data_reduced.max() - data_reduced.min()
                )
                colors_255 = (data_reduced * 255).astype(np.uint8)
                V = batch["vertices"][0].cpu().numpy()  # noqa: N806
                F = batch["faces"][0].cpu().numpy()  # noqa: N806
                if self.cfg.vertex_feature:
                    colored_mesh = trimesh.Trimesh(
                        vertices=V, faces=F, vertex_colors=colors_255, process=False
                    )
                else:
                    colored_mesh = trimesh.Trimesh(
                        vertices=V, faces=F, face_colors=colors_255, process=False
                    )
                colored_mesh.export(f"{save_dir}/feat_pca_{uid}_{view_id}.ply")
                ############
                torch.cuda.empty_cache()

            else:
                ### Mesh input (obj file)
                V = batch["vertices"][0].cpu().numpy()  # noqa: N806
                F = batch["faces"][0].cpu().numpy()  # noqa: N806

                ##### Loop through faces #####
                num_samples_per_face = self.cfg.n_point_per_face

                all_point_feats = []
                for face in F:
                    # Get the vertices of the current face
                    v0, v1, v2 = V[face]

                    # Generate random barycentric coordinates
                    u = np.random.rand(num_samples_per_face, 1)
                    v = np.random.rand(num_samples_per_face, 1)
                    is_prob = (u + v) > 1
                    u[is_prob] = 1 - u[is_prob]
                    v[is_prob] = 1 - v[is_prob]
                    w = 1 - u - v

                    # Calculate points in Cartesian coordinates
                    points = u * v0 + v * v1 + w * v2

                    tensor_vertices = (
                        torch.from_numpy(points.copy()).reshape(1, -1, 3).cuda().to(torch.float32)
                    )
                    point_feat = sample_triplane_feat(part_planes, tensor_vertices)  # N, M, C

                    #### Take mean feature in the triangle
                    point_feat = torch.mean(point_feat, axis=1).cpu().detach().numpy()
                    all_point_feats.append(point_feat)
                ##############################

                all_point_feats = np.array(all_point_feats).reshape(-1, 448)

                point_feat = all_point_feats

                np.save(f"{save_dir}/part_feat_{uid}_{view_id}.npy", point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                from sklearn.decomposition import PCA

                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (
                    data_reduced.max() - data_reduced.min()
                )
                colors_255 = (data_reduced * 255).astype(np.uint8)

                colored_mesh = trimesh.Trimesh(
                    vertices=V, faces=F, face_colors=colors_255, process=False
                )
                colored_mesh.export(f"{save_dir}/feat_pca_{uid}_{view_id}.ply")
                ############

        print("Time elapsed: " + str(time.time() - starttime))

        return


def download_partfield_checkpoint(cache_dir=None, force_download=False):
    repo_id = "mikaelaangel/partfield-ckpt"
    filename = "model_objaverse.ckpt"

    log.info(f"Downloading PartField checkpoint from {repo_id}")

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        log.info(f"Successfully downloaded checkpoint to: {checkpoint_path}")
        # convert checkpoint
        weights_only_checkpoint_path = checkpoint_path.replace(".ckpt", "_weights_only.ckpt")
        if os.path.exists(weights_only_checkpoint_path):
            return weights_only_checkpoint_path

        log.info(f"Converting checkpoint to weights only: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        torch.save(state_dict, weights_only_checkpoint_path)
        return weights_only_checkpoint_path
    except Exception as e:
        log.error(f"Failed to download checkpoint: {e}")
        raise RuntimeError(f"Could not download PartField checkpoint from {repo_id}") from e


def load_partfield_checkpoint(model, checkpoint_path, strict=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading PartField checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict)

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        log.info("Successfully loaded PartField checkpoint")

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "checkpoint_info": (checkpoint.get("info", {}) if isinstance(checkpoint, dict) else {}),
        }

    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}") from e


def create_partfield(config, pretrained=True, checkpoint_path=None):
    model = Model(config)
    if pretrained:
        if checkpoint_path is None:
            checkpoint_path = download_partfield_checkpoint()

        load_partfield_checkpoint(model, checkpoint_path)
    return model
