import json
from easydict import EasyDict
import os
import datetime

MAX_EPOCH = 100
NUM_LABELS = {'bace': 12, 'bbbp': 12, 'clintox': 12, 'hiv': 12, 'muv': 12, 'sider': 12, 'tox21': 12, 'toxcast': 12}
NUM_CLASSES = {'bace': 1, 'bbbp': 1, 'clintox': 2, 'hiv': 1, 'muv': 17, 'sider': 27, 'tox21': 12, 'toxcast': 617}
LEARNING_RATES = {'bace': 1e-4, 'bbbp': 1e-4, 'clintox': 1e-4, 'hiv': 1e-4, 'muv': 1e-4, 'sider': 1e-4, 'tox21': 1e-4, 'toxcast': 1e-4}
DECAY_RATES = {'bace': 0.3, 'bbbp': 0.5, 'clintox': 0.5, 'hiv': 0.5, 'muv':0.5, 'sider':0.5, 'tox21': 0.5, 'toxcast': 0.5}
CHOSEN_EPOCH = {'bace': MAX_EPOCH, 'bbbp': MAX_EPOCH, 'clintox': MAX_EPOCH, 'hiv': MAX_EPOCH, 'muv': MAX_EPOCH, 'sider': MAX_EPOCH, 'tox21': MAX_EPOCH, 'toxcast': MAX_EPOCH}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    
    # parse the configurations from the config json file provided
    
    with open(json_file, 'r') as config_file:
        
        config_dict = json.load(config_file)
    
    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.num_classes = NUM_CLASSES[config.dataset_name]
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("./experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.parent_dir, "checkpoint/")
    
    config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
    config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
    config.hyperparams.decay_rate = DECAY_RATES[config.dataset_name]
    
    return config
