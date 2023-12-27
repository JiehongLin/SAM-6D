import torch
import logging
import numpy as np


def load_checkpoint(model, checkpoint_path, checkpoint_key=None, prefix=""):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint_key is not None:
        pretrained_dict = checkpoint[checkpoint_key]  # "state_dict"
    else:
        pretrained_dict = checkpoint
    pretrained_dict = {k.replace(prefix, ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()
    # compare keys and update value
    pretrained_dict_can_load = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    pretrained_dict_cannot_load = [
        k for k, v in pretrained_dict.items() if k not in model_dict
    ]
    model_dict_not_update = [
        k for k, v in model_dict.items() if k not in pretrained_dict
    ]
    module_cannot_load = np.unique(
        [k.split(".")[0] for k in pretrained_dict_cannot_load]  #
    )
    module_not_update = np.unique([k.split(".")[0] for k in model_dict_not_update])  #
    logging.info(f"Cannot load: {module_cannot_load}")
    logging.info(f"Not update: {module_not_update}")
    logging.info(
        f"Pretrained: {len(pretrained_dict)}/ Loaded: {len(pretrained_dict_can_load)}/ Cannot loaded: {len(pretrained_dict_cannot_load)} VS Current model: {len(model_dict)}"
    )
    model_dict.update(pretrained_dict_can_load)
    model.load_state_dict(model_dict)
    logging.info("Load pretrained done!")
