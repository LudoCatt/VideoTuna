import functools
import json
import logging
import operator
import os
from typing import Tuple

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.cluster import DistCoordinator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url

from .misc import get_logger

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": hf_endpoint
    + "/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
    "PixArt-Sigma-XL-2-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-256x256.pth",
    "PixArt-Sigma-XL-2-512-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-512-MS.pth",
    "PixArt-Sigma-XL-2-1024-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-1024-MS.pth",
    "PixArt-Sigma-XL-2-2K-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-2K-MS.pth",
}


def reparameter(ckpt, name=None, model=None):
    model_name = name
    name = os.path.basename(name)
    if not dist.is_initialized() or dist.get_rank() == 0:
        get_logger().info("loading pretrained model: %s", model_name)
    if name in ["DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"]:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    if name in ["Latte-XL-2-256x256-ucf101.pt"]:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    if name in [
        "PixArt-XL-2-256x256.pth",
        "PixArt-XL-2-SAM-256x256.pth",
        "PixArt-XL-2-512x512.pth",
        "PixArt-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-256x256.pth",
        "PixArt-Sigma-XL-2-512-MS.pth",
        "PixArt-Sigma-XL-2-1024-MS.pth",
        "PixArt-Sigma-XL-2-2K-MS.pth",
    ]:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    if name in [
        "PixArt-1B-2.pth",
    ]:
        ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]

    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if (
            ckpt["y_embedder.y_embedding"].shape[0]
            < model.y_embedder.y_embedding.shape[0]
        ):
            get_logger().info(
                "Extend y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            additional_length = (
                model.y_embedder.y_embedding.shape[0]
                - ckpt["y_embedder.y_embedding"].shape[0]
            )
            new_y_embedding = torch.zeros(
                additional_length, model.y_embedder.y_embedding.shape[1]
            )
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat(
                [ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0
            )
        elif (
            ckpt["y_embedder.y_embedding"].shape[0]
            > model.y_embedder.y_embedding.shape[0]
        ):
            get_logger().info(
                "Shrink y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            ckpt["y_embedder.y_embedding"] = ckpt["y_embedder.y_embedding"][
                : model.y_embedder.y_embedding.shape[0]
            ]
    # stdit3 special case
    if type(model).__name__ == "STDiT3" and "PixArt-Sigma" in name:
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            if "blocks." in key:
                ckpt[key.replace("blocks.", "spatial_blocks.")] = ckpt[key]
                del ckpt[key]

    return ckpt


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model_ckpt = download_model(model_name)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    elif os.path.isfile(model_name) and model_name.split("/")[-1] in pretrained_models:
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name.split("/")[-1], model=model)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(
            model_name
        ), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def download_model(model_name=None, local_path=None, url=None):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if model_name is not None:
        assert model_name in pretrained_models
        local_path = f"pretrained_models/{model_name}"
        web_path = pretrained_models[model_name]
    else:
        assert local_path is not None
        assert url is not None
        web_path = url
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        dir_name = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        download_url(web_path, dir_name, file_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_from_sharded_state_dict(model, ckpt_path, model_name="model", strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)


def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(
                param.data.view(-1), [0, padding_size]
            )
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(
                model_shape_dict[name]
            )
    dist.barrier()


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def load_checkpoint(
    model, ckpt_path, save_as_pt=False, model_name="model", strict=False
):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path, model=model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path, model_name, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            get_logger().info("Model checkpoint saved to %s", save_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def save_frequently(
    booster: Booster,
    model: nn.Module,
    ema: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    global_step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
    shape_dict: dict,
):
    save_dir = os.path.join(save_dir, f"last")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    # ema is not boosted, so we don't need to use booster.save_model
    model_gathering(ema, shape_dict)
    global_rank = dist.get_rank()
    if int(global_rank) == 0:
        torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))
        model_sharding(ema)

    booster.save_optimizer(
        optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096
    )
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))
    dist.barrier()


def save(
    booster: Booster,
    save_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
    epoch: int = None,
    step: int = None,
    global_step: int = None,
    batch_size: int = None,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    if model is not None:
        booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    if optimizer is not None:
        booster.save_optimizer(
            optimizer,
            os.path.join(save_dir, "optimizer"),
            shard=True,
            size_per_shard=4096,
        )
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    if dist.get_rank() == 0:
        running_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "batch_size": batch_size,
            "sample_start_index": step * batch_size,
        }
        save_json(running_states, os.path.join(save_dir, "running_states.json"))

        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))

        if sampler is not None:
            # only for VariableVideoBatchSampler
            torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))
    dist.barrier()
    return save_dir


def load(
    booster: Booster,
    load_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
) -> Tuple[int, int, int]:
    assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
    assert os.path.exists(
        os.path.join(load_dir, "running_states.json")
    ), "running_states.json does not exist"
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    if model is not None:
        booster.load_model(model, os.path.join(load_dir, "model"))
    if ema is not None:
        # ema is not boosted, so we don't use booster.load_model
        ema.load_state_dict(
            torch.load(
                os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu")
            ),
            strict=False,
        )
    if optimizer is not None:
        booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))
    dist.barrier()

    if "sample_start_index" not in running_states:
        return (
            running_states["epoch"],
            running_states["step"],
        )
    else:
        return (
            running_states["epoch"],
            running_states["step"],
            running_states["sample_start_index"],
        )


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
