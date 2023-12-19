import copy
import math
import os
import os.path
import random

import loguru
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from osgeo import gdal
from . import augmentation_tif as psp_trsform
# from . import augmentation as psp_trsform
from .base import BaseDataset


# from .base import BaseDataset
class tif_dset(BaseDataset):
    def __init__(
            self, data_root, data_list, trs_form, mode='label', in_channels=3, num_classes=21):
        self.mode = mode
        # super(tif_dset, self).__init__(data_list, data_type)
        self.data_root = data_root
        # self.mode = mode
        self.transform = trs_form
        self.list_sample_new = data_list
        self.in_channels = in_channels
        self.num_classes=num_classes
        # # random.seed(seed)
        # if split == "train" and len(self.list_sample) > n_sup:
        #     self.list_sample_new = random.sample(self.list_sample, n_sup)
        # elif split == "train" and len(self.list_sample) < n_sup:
        #     num_repeat = math.ceil(n_sup / len(self.list_sample))
        #     self.list_sample = self.list_sample * num_repeat
        #
        #     self.list_sample_new = random.sample(self.list_sample, n_sup)
        # else:
        #     self.list_sample_new = self.list_sample
        #
        # del self.list_sample  # del掉原始list，減少無用數據

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, 'jpg', self.list_sample_new[index][0])
        if self.in_channels == 4:
            image_tif = gdal.Open(image_path).ReadAsArray()
            # scale to 255
            image_rgb = image_tif[:3]
            image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min()) * 255
            image_nir = image_tif[3]
            image_nir = (image_nir - image_nir.min()) / (image_nir.max() - image_nir.min()) * 255
            image = Image.fromarray(image_rgb.transpose(1, 2, 0), mode="RGB")
            image_nir = Image.fromarray(image_nir, mode="L")
        else:
            image = Image.open(image_path)
            image_nir = None
        if self.mode == 'label':  # 無標簽數據生成全0mask
            label_path = os.path.join(self.data_root, 'anno', self.list_sample_new[index][1])
            label = Image.open(label_path)
        else:
            label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
        image, label, img_nir = self.transform(image, label, image_nir)
        if image_nir is not None:
            image = torch.cat((image, img_nir), dim=1)
        label = label[0, 0].long()
        return image[0], label, torch.nn.functional.one_hot(label, num_classes=self.num_classes+1)

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(input_shape, convert_map, num_classes):
    # if cfg["saver"]["task_name"]=="guangxi":
    #     from . import augmentation_label_convert as psp_trsform
    trs_form = []
    # mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    # if cfg.get("ColorJitter", False) and cfg["ColorJitter"]:
    trs_form.append(psp_trsform.RandomColorJitter())
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.NormalizeT())
    trs_form.append(psp_trsform.ConvertLabel(num_classes, convert_map))
    # if cfg.get("resize", False):
    #     trs_form.append(psp_trsform.Resize(cfg["resize"]))
    # if cfg.get("rand_resize", False):
    trs_form.append(psp_trsform.RandResize(scale=[0.5, 2.0]))
    # if cfg.get("rand_rotation", False):
    #     rand_rotation = cfg["rand_rotation"]
    #     trs_form.append(
    #         psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
    #     )
    # if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
    #     trs_form.append(psp_trsform.RandomGaussianBlur())
    # if cfg.get("flip", False) and cfg.get("flip"):
    trs_form.append(psp_trsform.RandomHorizontalFlip())
    trs_form.append(psp_trsform.RandomVerticalFlip())
    # if cfg.get("crop", False):
    crop_size, crop_type = input_shape, 'center'
    trs_form.append(
        psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=255)
    )
    return psp_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = tif_dset(cfg["data_root"], os.path.join(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"])),
                    trs_form=trs_form, seed=seed, n_sup=n_sup)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


# build_costum_tif_loader(train_lines, input_shape, num_classes, True, VOCdevkit_path,
#                         convert_map)
def build_costum_tif_loader(train_lines, input_shape, num_classes, flag, VOCdevkit_path, convert_map, in_channels):
    # cfg_dset = all_cfg["dataset"]
    #
    # cfg = copy.deepcopy(cfg_dset)
    # cfg.update(cfg.get(split, {}))

    # workers = cfg.get("workers", 2)
    # batch_size = cfg.get("batch_size", 1)
    # if split == "val":
    #     batch_size = int(batch_size * 1.5)
    # n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(input_shape, convert_map, num_classes)
    dset = tif_dset(VOCdevkit_path, train_lines, trs_form=trs_form, mode='label',
                    in_channels=in_channels, num_classes= num_classes)

    return dset
    # build sampler
    # sample = DistributedSampler(dset)

    # loader = DataLoader(
    #     dset,
    #     batch_size=batch_size,
    #     num_workers=workers,
    #     sampler=sample,
    #     shuffle=False,
    #     pin_memory=False,
    # )
    # return loader


def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.stack(images).type(torch.FloatTensor) #torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.stack(pngs) #torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.stack(seg_labels).type(torch.FloatTensor) #torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels


def build_voc_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = tif_dset(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"]), trs_form=trs_form, seed=seed,
                    n_sup=n_sup, split=split)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = os.path.join(cfg["data_root"], cfg["data_list"]).replace("labeled.txt", "unlabeled.txt")
        dset_unsup = tif_dset(
            cfg["data_root"], data_list_unsup, trs_form=trs_form_unsup, seed=seed, n_sup=n_sup, split=split,
            mode='unlabel'
        )

        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup


def build_costum_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)

    data_list = os.path.join(cfg["data_root"], cfg["data_list"])
    n_sup = 0
    if split != 'val':
        # build sampler for unlabeled set
        data_list_unsup = os.path.join(cfg["data_root"], cfg["unlabel_data_list"])
        with open(data_list_unsup, 'r') as f:
            lines = f.readlines()
            line_count_unsup = len(lines)
        with open(data_list, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
        n_sup = line_count if line_count_unsup < line_count else line_count_unsup
        loguru.logger.info(f"训练样本数量为 2 * {n_sup}")
        # .replace("labeled.txt", "unlabeled.txt")
        dset_unsup = tif_dset(
            cfg["data_root"], data_list_unsup, 'costum', trs_form_unsup, seed, n_sup, split=split, mode="unlabel"
        )
        # n_sup = len(dset_unsup.list_sample_new)  # 計算一下無標記圖像的數量，將標籤圖像重複採樣至於無標記圖像相同

    dset = tif_dset(cfg["data_root"], data_list, trs_form, seed, n_sup,
                    split)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup


def main():
    import torch
    import torchvision.transforms as f
    import yaml
    from PIL import Image
    cfg = yaml.load(open(r'E:\python\ZEV\U2PL\config\config_cos_sup_unet.yaml', "r"), Loader=yaml.Loader)["dataset"]
    cfg["data_root"] = r"E:\python\ZEV\U2PL\cloud"
    cfg.update(cfg.get('train', {}))
    trs_form = build_transfrom(cfg)
    dset = tif_dset(cfg["data_root"], os.path.join(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"])),
                    'costum', trs_form=trs_form, seed=0, mode='unlabel')
    # sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=1,
        num_workers=0,
        # sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    for i, batch in enumerate(loader):
        a = batch
        print()
        pass

    # img_path = tif_dset()
    # img = Image.open(img_path)
    # trans = f.ColorJitter(brightness=0.8)
    # image = trans(img)
    # image.show()


if __name__ == "__main__":
    main()
