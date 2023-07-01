import torch
import torch.nn.functional as F
import importlib
from logzero import logger
from model import GCN, GAT


def find_module(full_module_name):
    try:
        return importlib.import_module(full_module_name)
    except ImportError as exc:
        if not (full_module_name + ".").startswith(exc.name + "."):
            raise


def _load_model(in_feats, prob_maps, weight_file=None, cuda_dev=None, mode_nn="GCN"):
    if mode_nn == "GCN":
        model = GCN(
            in_feats,  # input
            128,  # dimensions in hidden layers
            prob_maps,  # probability maps
            20,  # hidden layers
            F.relu,
            0.3,
        )
    else:
        model = GAT(
            in_feats,  # input
            128,  # dimensions in hidden layers
            prob_maps,  # probability maps
            10,  # hidden layers
            F.relu,
        )
    if cuda_dev is not None:
        model = model.to(cuda_dev)

    if weight_file:
        if cuda_dev is not None:
            model.load_state_dict(torch.load(weight_file))
        else:
            model.load_state_dict(
                torch.load(weight_file, map_location=torch.device("cpu"))
            )

    return model


def _locked_log(lock, msg, loglevel):
    if lock:
        if loglevel == "DEBUG":
            with lock:
                logger.debug(msg)
        elif loglevel == "INFO":
            with lock:
                logger.info(msg)
        elif loglevel == "WARN":
            with lock:
                logger.warn(msg)
        elif loglevel == "ERROR":
            with lock:
                logger.error(msg)
        else:
            with lock:
                logger.error(
                    f"The following message was logged with unknown log-level {loglevel}:\n{msg}"
                )
    else:
        if loglevel == "DEBUG":
            logger.debug(msg)
        elif loglevel == "INFO":
            logger.info(msg)
        elif loglevel == "WARN":
            logger.warn(msg)
        elif loglevel == "ERROR":
            logger.error(msg)
        else:
            logger.error(
                f"The following message was logged with unknown log-level {loglevel}:\n{msg}"
            )
