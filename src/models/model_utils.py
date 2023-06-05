import os
import csv
import torch
from loguru import logger
from models.architectures import model_net, finetuneNetwork
from sklearn.metrics import f1_score, average_precision_score


class Utils:
    def __init__(self, args_data, args_train_test, args_result):
        self.args_train_test = args_train_test
        self.args_data = args_data
        self.args_result = args_result
        if not os.path.exists(args_result.logPath):
            os.makedirs(args_result.logPath)

        self.logname = args_result.logPath + f"/{args_train_test.mode}" + ".csv"
        logger.info(f"Saving the logs to: {self.logname}")
        if not os.path.exists(self.logname):
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

    def check_model(self, image_size=(16, 3, 64, 64), num_classes=20):
        logger.info("Checking the model on random input")
        label_size = (image_size[0], num_classes)
        self.batch_x = torch.randn(image_size).to(self.device)
        self.batch_y = torch.randint(low=0, high=2, size=label_size).to(self.device)
        model = self.get_model(num_classes=num_classes).to(self.device)
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
        logger.info(f"Using device: {device}")
        return device

    def get_model(self, num_classes):
        model = model_net(num_classes=num_classes)
        model = torch.nn.DataParallel(model)
        return model

    def checkpoint(self, model, optimizer, f1, epoch, best, max_files_to_keep):
        # Save checkpoint.
        f1 = round(f1, 2)
        logger.info("Saving checkpoint..")
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "epoch": epoch + 1,
            "f1": f1,
        }
        if self.args_train_test.mode == "pretrain":
            model_path = self.args_result.pretrained_models_path
        else:
            model_path = self.args_result.finetuned_models_path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if best:
            torch.save(
                state, f"{model_path}/{self.args_train_test.mode}_{f1}_{epoch}_best.pt"
            )
        else:
            torch.save(
                state, f"{model_path}/{self.args_train_test.mode}_{f1}_{epoch}.pt"
            )
        # Remove older files if more than max_files_to_keep exist
        checkpoints = sorted(
            [os.path.join(model_path, fname) for fname in os.listdir(model_path)],
            key=os.path.getmtime,
        )
        while len(checkpoints) > max_files_to_keep:
            os.remove(checkpoints.pop(0))

    def load_checkpoint(self, model, optimizer, filepath, num_classes):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        if self.args_train_test.mode == "finetune":
            logger.info("Loading model for finetuning")
            fine_network = finetuneNetwork(hparams=self.args_train_test.finetune_params)
            finetuned_model = fine_network.finetune_model(
                model=model, num_classes=num_classes
            )
            optimizer = fine_network.get_optimizer(model=finetuned_model)
            return finetuned_model, optimizer, 0

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        logger.info(f"Restarting model from checkpoint at epoch: {epoch}")
        return model, optimizer, epoch

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
