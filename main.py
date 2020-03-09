import torch
import numpy as np
from datetime import datetime

"""
How To:
Example for running from command line:
python <path_to>/ProvablyPowerfulGraphNetworks/main_scripts/main_10fold_experiment.py --config=configs/10fold_config.json --dataset_name=COLLAB
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
# project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(project_dir)
# os.chdir(project_dir)

from data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
import utils.doc_utils as doc_utils


def main(configfile):
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = process_config(configfile)

    except Exception as e:
        print("missing or invalid arguments {}".format(e))
        exit(0)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(100)
    np.random.seed(100)

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)
    for exp in range(1, config.num_exp + 1):
        print("Experiment num = {}\n".format(exp))
        data = DataGenerator(config, )  # create a data generator

        model_wrapper = ModelWrapper(config, data)  # create an instance of the model you want

        trainer = Trainer(model_wrapper, data, config)  # create trainer and pass all the previous components to it

        trainer.train()  # here you train your model
        trainer.test()  # here you test your model


if __name__ == '__main__':
    start = datetime.now()

    main('config.json')
    print('Runtime: {}'.format(datetime.now() - start))
