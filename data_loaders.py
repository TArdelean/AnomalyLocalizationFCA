from pathlib import Path

import torch
import tifffile as tiff
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from evaluation import evaluate_experiment


class StandardFormatDataset(Dataset):
    def __init__(self, data_dir, out_dir, objects, paddings=(100,), pro_integration_limit=0.3, resize=False):
        super(StandardFormatDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.objects = objects

        self.paddings = paddings
        self.resize = resize
        self.pro_integration_limit = pro_integration_limit

        self.in_image_paths = self.get_input_paths()

    def __getitem__(self, index):
        in_image_path = self.in_image_paths[index]
        return in_image_path

    def __len__(self):
        return len(self.in_image_paths)

    def get_input_paths(self):
        in_image_paths = []
        for obj in self.objects:
            in_obj_path = self.data_dir / obj / "test"
            for defect in sorted(in_obj_path.iterdir()):
                in_image_paths.extend(sorted(list(defect.iterdir())))
        return in_image_paths

    def save_output(self, age: torch.Tensor, in_image_path):
        np_age = age.cpu().numpy()
        obj, _, defect, _ = in_image_path.parts[-4:]
        stem = in_image_path.stem
        out_def_path = self.out_dir / obj / "test" / defect
        vis_def_path = self.out_dir / obj / "visualize" / defect
        out_def_path.mkdir(parents=True, exist_ok=True)
        vis_def_path.mkdir(parents=True, exist_ok=True)

        tiff.imsave(out_def_path / f"{stem}.tiff", np_age)
        # Save for visualization
        vis = ((np_age - np_age.min()) / (np_age.max() - np_age.min()) * 255)
        Image.fromarray(vis.astype(np.uint8)).save(vis_def_path / f"{stem}.jpg")

    def run_evaluation(self):
        args = {
            'dataset_base_dir': str(self.data_dir),
            'anomaly_maps_dir': str(self.out_dir),
            'output_dir': str(self.out_dir),
            'evaluated_objects': self.objects,
            'pro_integration_limit': self.pro_integration_limit,
            'resize': self.resize if self.resize else 0,
        }
        args = type('dict_as_obj', (object,), args)  # Object from dict
        for padding in self.paddings:
            args.padding = padding
            evaluate_experiment.main(args)
