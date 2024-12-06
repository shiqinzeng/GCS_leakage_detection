# GCS leakage detection

Official PyTorch implementation <br>
CNN models for C02 leakage detection <br>


## Requirements

Python libraries: See [requirements.yml](requirements.yml) for library dependencies. The conda environment can be set up using these commands:

```.bash
conda env create -f requirements.yml
conda activate leakage_detection
```

## Training Process
Put the data under the [data](data/) directory, and
train the dataset by running the Python script.
```.bash
!python scripts/training_loop.py --dataset_path "data/dataset_jrm_1971_seismic_images?dl=0" --data_length 1971 --model_name "vgg16"
```
More details can be found under the notebook [training_demo.ipynb](scripts/training_demo.ipynb)

## Uncertainty Analysis
The multi-criteria decision-making (MCDM)-based Ensemble schema and uncertainty analysis details can be found on [artifacts_demo.ipynb](scripts/artifacts_demo.ipynb)
Below is the workflow diagram illustrating the process:

![Workflow Diagram](workflows/workflows.png)

[workflows.png](workflows/workflows.png)

