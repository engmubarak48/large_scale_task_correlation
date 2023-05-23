import csv
import wandb
from loguru import logger
import torch
import pdb
import numpy as np
from models.model_utils import Utils
from torch.autograd import Variable


class train_evaluate:
    def __init__(self, args_data, args_train_test, args_result):
        self.args_data = args_data
        self.args_train_test = args_train_test
        self.args_result = args_result
        self.best_f1 = float("-inf")
        self.model_utils = Utils(
            args_data=args_data,
            args_train_test=args_train_test,
            args_result=args_result,
        )
        self.model = self.model_utils.get_model()
        self.device = self.model_utils.device
        self.optimizer = self.model_utils.get_optimizer(self.model)
        self.criterion = self.model_utils.get_criterion()

    def _repeat(self, dataset):
        if self.args_train_test.dataset_name == "celebA":
            logger.info("repeating the dataset")
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args_train_test.batch_size
            )
        else:
            loader = dataset
        return loader

    def train(self, model, trainset, epoch, criterion, optimizer):
        logger.info("\nEpoch: %d" % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        predicted_labels = []
        all_targets = []
        trainloader = self._repeat(trainset)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets.squeeze())
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            all_targets.append(targets.detach().cpu())
            predicted_labels.append(predicted.detach().cpu())
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if batch_idx % self.args_train_test.train_batch_jump == 0:
                print(
                    batch_idx,
                    trainloader.dataset.nb_batches
                    if self.args_train_test.dataset_name == "celebA"
                    else len(trainloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / (total * self.args_data.num_classes),
                        correct,
                        (total * self.args_data.num_classes),
                    ),
                )

        all_targets = np.concatenate(all_targets)
        predicted_labels = np.concatenate(predicted_labels)
        AP, f1 = self.model_utils.scores(all_targets, predicted_labels)
        return (
            train_loss / batch_idx,
            100.0 * correct / (total * self.args_data.num_classes),
            100.0 * AP,
            100.0 * f1,
        )

    def test(self, model, validset, criterion, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_targets = []
        predicted_labels = []
        testloader = self._repeat(validset)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.squeeze().to(
                    self.device
                )

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())
                test_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                all_targets.append(targets.detach().cpu())
                predicted_labels.append(predicted.detach().cpu())
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                if batch_idx % self.args_train_test.test_batch_jump == 0:
                    print(
                        batch_idx,
                        testloader.dataset.nb_batches
                        if self.args_train_test.dataset_name == "celebA"
                        else len(testloader),
                        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                        % (
                            test_loss / (batch_idx + 1),
                            100.0 * correct / (total * self.args_data.num_classes),
                            correct,
                            (total * self.args_data.num_classes),
                        ),
                    )

        # Save checkpoint.
        all_targets = np.concatenate(all_targets)
        predicted_labels = np.concatenate(predicted_labels)
        AP, f1 = self.model_utils.scores(all_targets, predicted_labels)

        if f1 > self.best_f1:
            self.best_f1 = f1
            if self.args_result.save_ckpts:
                self.model_utils.checkpoint(model, f1, epoch)
        return (
            test_loss / batch_idx,
            100.0 * correct / (total * self.args_data.num_classes),
            100.0 * AP,
            100.0 * f1,
        )

    def train_eval(self, trainset, validset):
        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model
        for epoch in range(self.args_train_test.epochs):
            self.model_utils.adjust_learning_rate(optimizer, epoch)
            train_loss, train_acc, train_AP, train_f1 = self.train(
                model=model,
                trainset=trainset,
                epoch=epoch,
                criterion=criterion,
                optimizer=optimizer,
            )
            valid_loss, valid_acc, valid_AP, valid_f1 = self.test(
                model=model,
                validset=validset,
                criterion=criterion,
                epoch=epoch,
            )
            with open(self.model_utils.logname, "a") as logfile:
                logwriter = csv.writer(logfile, delimiter=",")
                logwriter.writerow(
                    [
                        epoch,
                        train_loss,
                        train_acc.item(),
                        valid_loss,
                        valid_acc.item(),
                        train_f1.item(),
                        valid_f1.item(),
                        train_AP.item(),
                        valid_AP.item(),
                    ]
                )

            logger.info(
                f"Epoch: {epoch} | train acc: {np.round(train_acc.item(), 3)} | test acc: {np.round(valid_acc.item(), 3)}"
            )
            logger.info(
                f"Epoch: {epoch} | train loss: {np.round(train_loss, 3)} | test loss: {np.round(valid_loss, 3)}"
            )
            logger.info(
                f"Epoch: {epoch} | train AP: {np.round(train_AP.item(), 3)} | test AP: {np.round(valid_AP.item(), 3)}"
            )
            logger.info(
                f"Epoch: {epoch} | train F1: {np.round(train_f1.item(), 3)} | test F1: {np.round(valid_f1.item(), 3)}"
            )

            if self.args_result.wandb_log:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "train_acc": train_acc.item(),
                        "valid_acc": valid_acc.item(),
                        "train_f1": train_f1.item(),
                        "valid_f1": valid_f1.item(),
                        "train_AP": train_AP.item(),
                        "valid_AP": valid_AP.item(),
                    }
                )
