# DIM-for-SVCT-recon

We have currently uploaded the code based on AAPM simulated dataset, in which the sparse sinogram has a resolution of 32 × 256, the full sinogram has a resolution of 512 × 256, and the corresponding CT image has a resolution of 256 × 256.


Before running the code, in addition to the commonly used deep learning libraries, you also need to install the following dependencies:

* [MONAI](https://github.com/Project-MONAI/MONAI)
* [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
* [torch-radon](https://github.com/matteo-ronchetti/torch-radon)

> **Note:** When installing `torch-radon`, you may need to apply a patch provided by [helix2fan](https://github.com/faebstn96/helix2fan) in order to install it properly.

To train the models at all levels, simply run:

```bash
python run.py
```

**run.py** is a comprehensive version that can train and validate all resolution levels of DIM. Details on parameter settings will be provided once the paper is accepted.
