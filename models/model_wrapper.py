import os
import torch
import torch.nn as nn
from models.base_model import BaseModel


class ModelWrapper(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data  # data loader
        self.model = BaseModel(config).to(config.device)
        self.loss = nn.BCEWithLogitsLoss(reduction="sum")

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, best: bool, epoch: int, optimizer: torch.optim.Optimizer):
        filename = 'best.tar' if best else 'last.tar'
        print("Saving model as {}...".format(filename), end=' ')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(self.config.checkpoint_dir, filename))
        print("Model saved.")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, best: bool):
        """
        :param best: boolean, to load best model or last checkpoint
        :return: tuple of optimizer_state_dict, epoch
        """
        # self.model = models.base_model.BaseModel(self.config).to('cuda')
        filename = 'best.tar' if best else 'last.tar'
        print("Loading {}...".format(filename), end=' ')
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(torch.device('cuda'))
        print("Model loaded.")

        return checkpoint['optimizer_state_dict'], checkpoint['epoch']

    def loss_and_results(self, scores, labels):
        loss = self.loss(scores, labels.float())
        pred_scores = torch.sigmoid(scores).detach().cpu()
        # correct_predictions = torch.eq(torch.argmax(scores, dim=1), labels).sum().cpu().item()
        correct_predictions = torch.eq((pred_scores > 0.5).clone().detach().float(), labels.cpu()).sum().cpu().item()

        return loss, correct_predictions, pred_scores.numpy()

    def run_model_get_loss_and_results(self, input, labels):
        result = self.loss_and_results(self.model(input), labels)
        return result

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
