import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as albu
from lib.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import cv2
import numpy as np
import random


class ImageTransform():
    def __init__(self):
        self.data_transform = {'train': Compose([
            # Scale(scale=[0.5, 1.5]),
            # RandomRotation(angle=[-10, 10]),
        ]),
            'val': Compose([])}

    def __call__(self, img, mask, phase='train'):
        return self.data_transform[phase](img, mask)


def get_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # albu.Rotate(limit=[-10, 10], p=1.0),
        # albu.ShiftScaleRotate(shift_limit=[-0.0625, 0.0625], scale_limit=[-0.1, 0.1], rotate_limit=[-30, 30],
        #                       interpolation=1, border_mode=4, value=None, mask_value=None, p=0.5),
        # albu.RandomGamma(gamma_limit=[50, 150], p=1.0),
        # albu.RandomSizedCrop([300, 400], 416, 416, p=0.5),
        # albu.RandomGridShuffle(grid=(2, 2), p=0.5),
        # albu.RandomBrightness(limit=0.2,p=0.5),
        # albu.RandomContrast(limit=0.2, p=0.5),
        # albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=0.5),
        # albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        # albu.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
        #                    fill_value=0, p=0.5)
    ]
    return albu.Compose(train_transform)


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize, phase='train'):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        self._alpha = 0.2

        self.transform = ImageTransform()
        self.transform2 = get_augmentation()
        self.phase = phase

        self.transform3 = albu.Compose(
            [
                albu.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                albu.ColorJitter(),
                albu.HorizontalFlip(),
            ]
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.proj = nn.Conv2d(3, 3, kernel_size=7, strid=7)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        print(f"{gt.shape=}")
        if self.phase == 'train':
            # if random.random() < 1.0:
            #     image, gt = self._apply_mixup(image, gt, index)

            # if random.random() < 1.0:
            #     image, gt = self.cutmix(image, gt, index)

            image = np.array(image)
            gt = np.array(gt)

            # image, gt = self.augment_and_mix(image, gt)

            # augmented = self.transform2(image=image, mask=gt)
            augmented = self.transform3(image=image, mask=gt)

            image, gt = augmented['image'], augmented['mask']
            image = Image.fromarray(image)
            gt = Image.fromarray(gt)
            image = image.convert('RGB')
            gt = gt.convert('L')

        # image, gt = self.transform(image, gt, phase=self.phase)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # TODO: パッチに分割する
        gt_patch = self.proj(gt)
        print(f"{gt_patch.shape=}")

        return image, gt_patch

    def _apply_mixup(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)
        # 混ぜる割合を決めて
        # r = np.random.beta(self._alpha, self._alpha, 1)[0]

        r = np.random.normal(loc=0, scale=3, size=1)
        r = 1 / (1 + np.exp(-r))
        # 画像、ラベルを混ぜる（クリップしないと範囲外になることがある）
        mixed_image = np.clip(r * image1 + (1 - r) * image2, 0, 255)
        mixed_gt = np.clip(r * gt1 + (1 - r) * gt2, 0, 255)
        mixed_image = Image.fromarray(mixed_image.astype(np.uint8))
        mixed_gt = Image.fromarray(mixed_gt.astype(np.uint8))
        mixed_image = mixed_image.convert('RGB')
        mixed_gt = mixed_gt.convert('L')
        return mixed_image, mixed_gt

    # Datasetの__get_item__のidx以外のindexを取得する
    def _get_pair_index(self, idx):
        r = list(range(0, idx)) + list(range(idx + 1, len(self.images)))
        return random.choice(r)

    def cutmix(self, image1, gt1, idx1):
        image1 = np.array(image1)
        gt1 = np.array(gt1)
        # mixする画像のインデックスを拾ってくる
        idx2 = self._get_pair_index(idx1)
        # 画像の準備
        image2 = self.rgb_loader(self.images[idx2])
        # ラベルの準備（アノテーションファイルは1,0,0,0のように所属するクラスが記されている）
        gt2 = self.binary_loader(self.gts[idx2])
        image2 = np.array(image2)
        gt2 = np.array(gt2)

        lam = np.random.beta(self._alpha, self._alpha)

        image_h, image_w, _ = image1.shape
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        image1[y0:y1, x0:x1, :] = image2[y0:y1, x0:x1, :]
        gt1[y0:y1, x0:x1] = gt2[y0:y1, x0:x1]

        return image1, gt1

    def augment_and_mix(self, image, gt, width=3, depth=-1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
          image: Raw input image as float32 np.ndarray of shape (h, w, c)
          severity: Severity of underlying augmentation operators (between 1 to 10).
          width: Width of augmentation chain
          depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
          alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        mix = np.zeros_like(image).astype('float32')
        mix_gt = np.zeros_like(gt).astype('float32')
        for i in range(width):
            image_aug = image.copy()
            gt_aug = gt.copy()
            d = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(self.augmentations)
                image_aug = op(image=image_aug)['image']
                gt_aug = op(image=gt_aug)['image']
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug
            mix_gt = ws[i] * gt_aug

        mixed = (1 - m) * image + m * mix
        mixed_gt = (1 - m) * gt + m * mix_gt
        return mixed.astype('uint8'), mixed_gt.astype('uint8')

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, phase='train',
               droplast=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, phase=phase)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=droplast)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def load_data_mixup(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        idx2 = self._get_pair_index(self.index)
        image2 = self.rgb_loader(self.images[idx2])
        image2 = self.transform(image2).unsqueeze(0)
        gt2 = self.binary_loader(self.gts[idx2])

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, image2, gt2, name

    def _get_pair_index(self, idx):
        r = list(range(0, idx)) + list(range(idx + 1, len(self.images)))
        return random.choice(r)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
