
import torch
import numpy as np
from PIL import Image
from coco.PythonAPI.pycocotools.coco import COCO
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CocoClassDatasetRandom:
    def __init__(self, images_path, annotation_path, pretrain = True, split_ratio = .5, transform = False):
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
        self.alice_labels, self.bob_labels = sorted(all_categories[:split_percent]), sorted(all_categories[split_percent:])
        alice_img_ids, bob_img_ids = [], []

        for ima_id in image_ids:
            al_img_cats = [catId for catId in self.alice_labels if len(self.coco.getAnnIds(imgIds=ima_id, catIds=[catId])) > 0]
            bob_img_cats = [catId for catId in self.bob_labels if len(self.coco.getAnnIds(imgIds=ima_id, catIds=[catId])) > 0]
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

        image_name = self.coco.loadImgs(ids=[index])[0]['file_name']
        image_path = f'{self.images_path}/{image_name}'
        labels = torch.tensor([1 if len(self.coco.getAnnIds(imgIds=index, catIds=[catId])) > 0 else 0 for catId in cat_ids]).unsqueeze(0)

        image = Image.open(image_path).resize((224,224))

        if self.transform:
            image = self.transform(image)

        return image, labels