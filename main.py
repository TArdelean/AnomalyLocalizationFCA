from typing import Callable

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from op_utils import load_image_tensor


def compute_anomaly_map(image: torch.tensor, feature_extractor: Callable, stat_method: Callable):
    features = feature_extractor(image)
    age = stat_method(features)
    return age


def run_anomaly_detection(dataset, cfg: DictConfig):
    fe = instantiate(cfg.fe.feature_extractor)
    sc = instantiate(cfg.sc.method)
    for in_image_path in tqdm(dataset):
        image = load_image_tensor(in_image_path, cfg.image_size, device=cfg.device)
        anomaly_map = compute_anomaly_map(image, fe, sc)

        dataset.save_output(anomaly_map, in_image_path)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    data_manager = instantiate(cfg.dataset.data_manager)
    run_anomaly_detection(data_manager, cfg)
    data_manager.run_evaluation()


if __name__ == "__main__":
    my_app()
