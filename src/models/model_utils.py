import os
import csv
import torch
from loguru import logger
from models.architectures import model_net
from sklearn.metrics import f1_score, average_precision_score


class Utils:
    def __init__(self, args_data, args_train_test, args_result):
        self.args_train_test = args_train_test
        self.args_data = args_data
        self.args_result = args_result
        if not os.path.exists(args_result.logPath):
            os.makedirs(args_result.logPath)

        self.logname = args_result.logPath + f"_{args_train_test.mode}" + ".csv"

        if not os.path.exists(self.logname):
            logger.info(f"creating metrics log file: {self.logname}")
            with open(self.logname, "w") as logfile:
                self.logwriter = csv.writer(logfile, delimiter=",")
                self.logwriter.writerow(
                    [
                        "epoch",
                        "train loss",
                        "train acc",
                        "test loss",
                        "test acc",
                        "train F1",
                        "test F1",
                    ]
                )
        self.device = self.get_device()
        self.check_model()

    def check_model(self, image_size=(16, 3, 64, 64), label_size=(16, 20)):
        self.batch_x = torch.randn(image_size).to(self.device)
        self.batch_y = torch.randint(low=0, high=2, size=label_size).to(self.device)
        logger.info("verifying if model is run on random data")
        model = self.get_model()
        logger.info("model: \n", model)
        self.outputs_ = model(self.batch_x)
        logger.info(
            f"Outputs shape: {self.outputs_.shape}, batch labels shape: {self.batch_y.shape}"
        )

    def get_criterion(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        logger.info(
            f"Initial random loss: {criterion(self.outputs_, self.batch_y.squeeze().float())}"
        )
        return criterion

    def get_optimizer(self, model):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args_train_test.lr,
            momentum=self.args_train_test.momentum,
            weight_decay=self.args_train_test.weight_decay,
        )
        return optimizer

    def scores(self, y_true, y_pred):
        AP = average_precision_score(y_true, y_pred, average="samples")
        f1 = f1_score(y_true, y_pred, average="samples")
        return AP, f1

    def get_device(self):
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logger.info(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                logger.info(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = torch.device("mps")
        return device

    def get_model(self):
        model = model_net(num_classes=self.args_data.num_classes)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        return model

    def checkpoint(self, model, f1, epoch):
        # Save checkpoint.
        logger.info("Saving checkpoint..")
        state = {
            "state_dict": model.state_dict(),
            "f1": f1,
            "epoch": epoch,
            "rng_state": torch.get_rng_state(),
        }
        if self.args_result.mode == "pretrain":
            model_path = self.args_result.pretrained_models_path
        else:
            model_path = self.args_result.finetuned_models_path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        torch.save(state, f"{model_path}/{self.args_train_test.mode}_{f1}_{epoch}.t7")

    def adjust_learning_rate(self, optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = self.args_train_test.lr
        if epoch <= 9 and lr > 0.1:
            # warm-up training for large minibatch
            lr = 0.1 + (self.args_train_test.lr - 0.1) * epoch / 10.0
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
