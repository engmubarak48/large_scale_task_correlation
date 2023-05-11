
import yaml
import wandb
from data_prep.datamodule import dataloading
from models.train_test import train_evaluate


def main(args):
    args_train_test = args.datamodule.args_train_test
    dataset_name = args_train_test.dataset_name
    args_data = args.datamodule["dataset_name"]
    args_result = args.args_result

    if args_result.wandb_log:
        wandb.init(project=args.project_name, entity="lsc")

    # get dataloader
    loader = dataloading(
        dataset_name=dataset_name, args_data=args_data, args_train_test=args_train_test
    )
    train_eval = train_evaluate(
        args_data=args_data, args_train_test=args_train_test, args_result=args_result
    )

    trainloader, valid_loader = loader.get_dataloaders()
    train_eval.train_eval(train_loader=trainloader, val_loader=valid_loader)


if __name__ == "__main__":
    yml_path = "./src/config.yaml"
    with open(yml_path, encoding="utf8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    main(args)
