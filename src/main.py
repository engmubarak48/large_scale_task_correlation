import pdb
import wandb
import yaml
from box import Box
from data_prep.datamodule import dataloading
from models.train_test import train_evaluate


def main(args):
    args_train_test = args.datamodule.args_train_test
    dataset_name = args_train_test.dataset_name
    args_data = args.datamodule.args_data[dataset_name]
    args_result = args.datamodule.args_result
    if args_result.wandb_log:
        wandb.init(project=args_result.project_name, entity="jmohamud")

    # get dataloader
    loader = dataloading(
        dataset_name=dataset_name, args_data=args_data, args_train_test=args_train_test
    )
    num_classes = loader.get_number_classes
    trainloader, validloader = loader.get_dataloaders()
    train_eval = train_evaluate(
        args_data=args_data,
        args_train_test=args_train_test,
        args_result=args_result,
        num_classes=num_classes,
    )
    train_eval.train_eval(trainset=trainloader, validset=validloader)


if __name__ == "__main__":
    yml_path = "./src/config.yaml"
    with open(yml_path, encoding="utf8") as f:
        args = Box(yaml.load(f, Loader=yaml.FullLoader))

    main(args)
