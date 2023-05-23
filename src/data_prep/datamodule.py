import torch
import numpy as np
import torchvision
from PIL import Image
import multiprocessing
from typing import Dict, Any
from cocoapi.PythonAPI.pycocotools.coco import COCO
import torchvision.transforms as transforms


class CocoClassDatasetRandom:
    def __init__(
        self,
        images_path,
        annotation_path,
        pretrain=True,
        split_ratio=0.5,
        transform=False,
    ):
        self.coco = COCO(annotation_path)
        self.cat_ids = self.coco.getCatIds()
        self.images_path = images_path
        self.transform = transform
        self.pretrain = pretrain
        self.split_ratio = split_ratio
        self.alice_images, self.bob_images = self.random_label_split()

    def random_label_split(self):
        all_categories = np.array(self.cat_ids)
        image_ids = np.array(self.coco.getImgIds())
        np.random.shuffle(all_categories)
        split_percent = int(self.split_ratio * len(all_categories))
        self.alice_labels, self.bob_labels = sorted(
            all_categories[:split_percent]
        ), sorted(all_categories[split_percent:])
        alice_img_ids, bob_img_ids = [], []

        for ima_id in image_ids:
            al_img_cats = [
                catId
                for catId in self.alice_labels
                if len(self.coco.getAnnIds(imgIds=ima_id, catIds=[catId])) > 0
            ]
            bob_img_cats = [
                catId
                for catId in self.bob_labels
                if len(self.coco.getAnnIds(imgIds=ima_id, catIds=[catId])) > 0
            ]
            if al_img_cats:
                alice_img_ids.append(ima_id)
            if bob_img_cats:
                bob_img_ids.append(ima_id)

        return np.array(alice_img_ids), np.array(bob_img_ids)

    def __len__(self):
        if self.pretrain:
            return len(self.alice_images)
        else:
            return len(self.bob_images)

    def __getitem__(self, idx):
        if self.pretrain:
            index = self.alice_images[idx]
            cat_ids = self.alice_labels
        else:
            index = self.bob_images[idx]
            cat_ids = self.bob_labels

        image_name = self.coco.loadImgs(ids=[index])[0]["file_name"]
        image_path = f"{self.images_path}/{image_name}"
        labels = torch.tensor(
            [
                1 if len(self.coco.getAnnIds(imgIds=index, catIds=[catId])) > 0 else 0
                for catId in cat_ids
            ]
        ).unsqueeze(0)

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, labels


class iterable_random_celebA_split(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path: str = "datasets/",
        split: str = "all",
        target_type: str = "attr",
        download: bool = True,
        mode="pretrain",
        batch_size=4,
        transform: callable = transforms.ToTensor(),
    ):
        super(iterable_random_celebA_split).__init__()
        self.celebData = torchvision.datasets.CelebA(
            root=data_path,
            split=split,
            target_type=target_type,
            transform=transform,
            download=download,
        )
        # remove empty label from label names.
        self.celebData.attr_names.remove("")
        assert len(self.celebData.attr_names) == 40, "There are more than 40 classes"
        self.mode = mode
        self.batch_size = batch_size
        self.used_idxs = set()
        if self.mode == "pretrain":
            self.labels = np.load(file=f"{data_path}/pretrain_labels.npy").tolist()
        else:
            self.labels = np.load(file=f"{data_path}/finetune_labels.npy").tolist()

    def __iter__(self):
        """Makes a single example generator of the loaded data."""
        idx = 0
        while True:
            # This will reset the index to 0 if we are at the end of the dataset.
            if idx == len(self.celebData):
                idx = idx % len(self.celebData)
                self.used_idxs = set()
                return
            image, image_labels, image_label_names = self.get_image_by_index(idx)
            if not bool(image_label_names & set(self.labels)):
                idx += 1
                continue
            if idx in self.used_idxs:
                raise Exception(
                    f"Index: {idx} is already used and len used_idxs: {len(self.used_idxs)} and len celebdata: {len(self.celebData)}"
                )
            self.used_idxs.add(idx)
            idx += 1
            image_labels = torch.tensor(
                [
                    1 if self.labels[i] in image_label_names else 0
                    for i in range(len(self.labels))
                ]
            )
            yield image, image_labels

    @property
    def nb_batches(self):
        return len(self.celebData) // self.batch_size

    def get_image_by_index(self, idx):
        image, image_labels = self.celebData[idx]
        # align image and labels
        image_label_names = {
            self.celebData.attr_names[i]
            for i in range(len(self.celebData.attr_names))
            if image_labels[i] == 1
        }
        return image, image_labels, image_label_names


class dataloading:
    def __init__(
        self,
        dataset_name: str,
        args_data: Dict[str, Any],
        args_train_test: Dict[str, Any],
    ):
        self.num_workers = multiprocessing.cpu_count()
        self.no_of_gpus = torch.cuda.device_count()
        self.dataset_name = dataset_name
        if dataset_name == "celebA":
            train_transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(args_data.resize_size, args_data.resize_size)
                    ),
                    # transforms.RandomCrop(resize_size, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        args_data.stats["mean"], args_data.stats["std"]
                    ),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(args_data.resize_size, args_data.resize_size)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        args_data.stats["mean"], args_data.stats["std"]
                    ),
                ]
            )
            self.celeb_train_data = iterable_random_celebA_split(
                data_path=args_data.train_images_path,
                split="train",
                mode=args_train_test.mode,
                batch_size=args_train_test.batch_size,
                transform=train_transform,
            )
            self.celeb_val_data = iterable_random_celebA_split(
                data_path=args_data.val_images_path,
                split="valid",
                mode=args_train_test.mode,
                batch_size=args_train_test.batch_size,
                transform=val_transform,
            )
        if dataset_name == "coco":
            train_transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(args_data.resize_size, args_data.resize_size)
                    ),
                    # transforms.RandomCrop(args_data.resize_size, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.change_to_3_channel,
                    transforms.Normalize(
                        args_data.stats["mean"], args_data.stats["std"]
                    ),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(args_data.resize_size, args_data.resize_size)
                    ),
                    transforms.ToTensor(),
                    self.change_to_3_channel,
                    transforms.Normalize(
                        args_data.stats["mean"], args_data.stats["std"]
                    ),
                ]
            )

            self.coco_train_dataset = CocoClassDatasetRandom(
                images_path=args_data.train_images_path,
                annotation_path=args_data.train_annotation_path,
                transform=train_transform,
            )
            self.coco_val_dataset = CocoClassDatasetRandom(
                images_path=args_data.val_images_path,
                annotation_path=args_data.val_annotation_path,
                transform=val_transform,
            )

    def change_to_3_channel(self, x):
        if x.size()[0] == 1:
            return x.repeat(3, 1, 1)
        return x

    def get_dataloaders(self):
        if self.dataset_name == "celebA":
            trainloader = self.celeb_train_data
            valloader = self.celeb_val_data
        elif self.dataset_name == "coco":
            trainloader = torch.utils.data.DataLoader(
                self.coco_train_dataset,
                batch_size=self.args_train_test.batch_size,
                shuffle=True,
            )  # , num_workers=args.num_workers)
            valloader = torch.utils.data.DataLoader(
                self.coco_val_dataset,
                batch_size=self.args_train_test.batch_size,
                shuffle=False,
            )  # , num_workers=args.num_workers)
        else:
            raise ValueError(f"Unknown Dataset: {self.dataset_name}")
        return trainloader, valloader
