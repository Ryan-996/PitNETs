# PitNETs

## Project

Classification of pituitary neuroendocrine tumors using H&E whole-slide images (WSIs), with WSI-only and WSI-clinical models.

**Patient data and pretrained weights are not provided.** Obtain Virchow2 access separately.

## Paper and citation

Zheng, A., Lin, R., Chen, C. et al. [A clinically conditional deep learning model for predicting histological subtypes of pituitary neuroendocrine tumors from hematoxylin-eosin whole-slide images](https://doi.org/10.1186/s12911-026-03918-y). *BMC Med Inform Decis Mak* (2026).

## Environment

Use the `pitnets` Conda environment with Python 3.11, PyTorch 2.7.0+cu128, and dependencies in [environment.yml](environment.yml). Feature extraction, training, and inference require CUDA.

From the repository root, create the environment with `conda env create -f environment.yml`, then activate it with `conda activate pitnets`.

## Workflow

Run modules from the repository root; use `--help` for arguments.

1. `python -m preprocessing.select_patches` — Select tissue patches.
2. `python -m preprocessing.augment` — Sample unused patch coordinates.
3. `python -m preprocessing.extract_features` — Extract Virchow2 CLS features.
4. `python -m training.train` — Train WSI-only or WSI-clinical models.
5. `python -m inference.predict` — Average softmax probabilities from five folds.
6. `python -m evaluation.evaluate` — Export metrics, confusion matrices, and ROC curves.


## License

Original code uses the [MIT license](LICENSE). Third-party libraries and Virchow2 retain their own licenses and access conditions.
