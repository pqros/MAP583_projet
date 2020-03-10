# MAP583_projet

This repository holds a PyTorch implementation of MAP583-Apprentissage profond(2019-2020) course project "How powerful are graph neural networks". We test model presented in paper [Provably powerful graph networks](https://arxiv.org/pdf/1905.11136.pdf) on the 8 molecule datasets (BBBP, BACE, CLINTOX, TOX21, TOXCAST, HIV, MUV, SIDER) in MoleculeNet. 

## Data
before running the code, the data should be processed in format of neural network input using following commands:
```
cd MAP583_projet
python rawdata_generator.py
```

## Code:
### Code structure
```
.
├── MAP583-soutenance.pdf
├── __pycache__
│   ├── data_generator.cpython-36.pyc
│   ├── helper.cpython-36.pyc
│   └── trainer.cpython-36.pyc
├── config.json
├── data
├── data_generator.py
├── experiments
├── helper.py
├── main.py
├── models
│   ├── __pycache__
│   │   ├── base_model.cpython-36.pyc
│   │   ├── layers.cpython-36.pyc
│   │   ├── model_wrapper.cpython-36.pyc
│   │   └── modules.cpython-36.pyc
│   ├── base_model.py
│   ├── layers.py
│   ├── model_wrapper.py
│   └── modules.py
├── rawdata
│   ├── bace.csv
│   ├── bbbp.csv
│   ├── clintox.csv
│   ├── hiv.csv
│   ├── muv.csv
│   ├── sider.csv
│   ├── splitters.py
│   ├── tox21.csv
│   ├── toxcast.csv
│   └── utils.py
├── rawdata_generator.py
├── trainer.py
└── utils
├── __pycache__
│   ├── config.cpython-36.pyc
│   ├── dirs.cpython-36.pyc
│   └── doc_utils.cpython-36.pyc
├── config.py
├── dirs.py
└── doc_utils.py
```
### Requirements:
```
torch 1.4.0
easydict
pandas
python3.7
```

### Configuration
To modify network architecture and test dataset, see `config.json`. To adjust hyper-parameters, see `utils/config.py`.

### Run the code
After configuration, run the command in the root directory: `python main.py`

## Group members
DONG Tian, GUO Shijia, GUO Yanzhu, NI Runbo
