# GCS  leakage detection

Official PyTorch implementation <br>
CNN models for C02 leakage detection <br>


## Requirements

Python libraries: See [requirements.yml](requirements.yml) for library dependencies. The conda environment can be set up using these commands:

```.bash
conda env create -f requirements.yml
conda activate leakage_detection
```

## Training Process
Put the data on 
train on the full dataset by running the Python script.
```.bash
!python scripts/training_loop.py --dataset_path "data/dataset_jrm_1971_seismic_images?dl=0" --data_length 1971 --model_name "vgg16"
```
More details can be found under the notebook [test.ipynb](scripts/training_demo.ipynb)
## Test the pretrained model

The pre-trained model is in the directory [pretrained_model](pretrained_model). See the details in the [test.ipynb](test.ipynb).



