# High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis (WACV 2024)

### [Project Page](https://reality.tf.fau.de/pub/ardelean2024highfidelity.html) | [Paper](https://arxiv.org/abs/2304.06433)

The official implementation of *High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis*.

![Teaser](static/teaser.png)

## Installation
We implemented our method using PyTorch. 
For an easy installation, we provide an environment file that contains all dependencies:

```
conda env create -f environment.yml
conda activate fca
```

## Demo
We include a minimal script that allows running our code on sample images to easily try our method.
By default, running the `demo.py` file will compute the anomaly score for the provided image example `example/000.png`.
```
python demo.py
```

## Data
To run the evaluation code, you have to first prepare the desired dataset.
Our data loader assumes the data follows the file structure of the MVTec anomaly detection dataset.
You can download the MVTec AD from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).
The woven fabric textures dataset (WFT) can be downloaded [here](https://www.mydrive.ch/shares/46066/8338a11f32bb1b7b215c5381abe54ebf/download/420939225-1629955758/textures.zip).
Please extract the data into the `datasets` directory.
To use any other dataset that follows the same file structure you can simply place the data in the same folder and create a corresponding dataset config file under `conf/dataset`. To understand the required format, please see `conf/dataset/mvtec.yaml`.

## Run and evaluate on a dataset
To evaluate the method, use the `main.py` script. We use Hydra to manage the command line interface, which makes it easy to specify the method and dataset configurations.
For example, running our FCA statistics comparison (`sc`) with WideResnet features (`fe`) on the MVTec dataset is done using
```
python main.py dataset=mvtec fe=wide sc=fca image_size=original tile_size=[9,9]
```
To run the same on the WFT dataset 
```
python main.py dataset=wft fe=wide sc=fca image_size=original tile_size=[9,9]
```

We automatically run the evaluation code after computing the anomaly scores for all images in the dataset.
You can inspect the metrics in the json file `outputs/{experiment_name}/metrics_{padding}.json` and visualize the predicted anomaly maps under `outputs/{experiment_name}/{object_name}/visualize`


## Content
This repository contains several options for feature extraction, as described in our paper: plain colors (`color`), random kernels (`random`), VGG-network (`vgg`), WideResnet-network (`wide`), steerable filters (`steerable`), and Laws' texture energy measure (`ltem`).

For patch statistics comparison, you can opt for one of the following: moments-based (`moments`), histogram-based (`hist`), sample weighted wasserstein (`sww`), feature correspondence analysis (`fca`), and our reimplementation of the method of Aota et al. (`aota`).

To reproduce one of our experiments for design space exploration (for example, histogram-based statistics comparison with VGG features) you can run:
```
python main.py dataset=mvtec fe=vgg sc=hist image_size=[256,256] tile_size=[25,25]
```

## Citation
Should you find our work useful in your research, please cite:
```BibTeX
@inproceedings{ardelean2023highfidelity,
    title = {High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis},
    author = {Ardelean, Andrei-Timotei and Weyrich, Tim},
    booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    numpages = 11,
    year = {2024},
    month = jan,
    day = 4,
    authorurl = {https://reality.tf.fau.de/pub/ardelean2024highfidelity.html},
}
```

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956585.

## License
Please see the [LICENSE](LICENSE).