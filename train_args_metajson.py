import argparse
import ast
import datetime
import os
import random
import sys
from functools import partial
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
# from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。
   如果格式有误，参考：https://github.com/bubbliiiing/segmentation-format-fix

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='vgg', choices=['vgg', 'resnet50'],
                        help='backbone')
    parser.add_argument('--classes', type=str, nargs='+', help='所有类别')
    parser.add_argument('--freeze', type=str, default='true', help='初始训练时，是否冻结backbone')
    parser.add_argument('--classes_weight', type=int, nargs='+', help='不同类别的权重,该类图像越少，数值越大')
    parser.add_argument('--categorys', type=int, nargs='+', help='需要预测的类别')
    parser.add_argument('--in_channels', type=int, default=3, help='输入图像波段数')
    parser.add_argument('--pretrained', type=str, default='true', help='是否使用预训练backbone权重')
    parser.add_argument('--transfer_path', type=str, default='', help='基于已有权重进行迁移学习的权重路径')
    parser.add_argument('--dataset_path', type=str, nargs='+', help='数据集路径')

    parser.add_argument('--training_epoch', type=int, default=6, help='整个网络训练的epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='整个网络全都更新的bz')

    parser.add_argument('--init_lr', type=float, default=1e-4, help='初始学习率')
    # parser.add_argument('--dice_loss', type=bool, default=True, help='是否使用dice loss')
    # parser.add_argument('--focal_loss', type=bool, default=True, help='是否使用focal loss,不使用则使用默认的CEloss')
    parser.add_argument('--dice_loss', type=str, default='true', help='是否使用dice loss')
    parser.add_argument('--focal_loss', type=str, default='true', help='是否使用focal loss,不使用则CEloss')
    parser.add_argument('--eval_epoch', type=int, default=1, help='验证频率，计算在验证集上的评价指标，会影响训练速度')
    parser.add_argument('--eval_metric', type=str, default='miou', help='评价指标')
    parser.add_argument('--weight_save_dir', type=str, default='log_2', help='权重及元数据保存路径')
    parser.add_argument('--log_save_dir', type=str, default='log_2/log.json', help='训练日志保存路径')
    # parser.add_argument('--model_param_save_dir', type=str, default='log_1/model_param.json', help='模型元数据保存路径')
    parser.add_argument('--providers', type=str, default='cuda', help='模型训练设备')

    args = parser.parse_args()
    return args


def write_filenames_to_file(file_names, file_path):
    with open(file_path, 'w') as file:
        for name in file_names:
            file.write(name + '\n')


# label的类别转化字典
def convert_label(CLASSES_need, classes_pri):
    # 想要的类别名称到类别值的映射
    desired_class_name_to_value = {v: k for k, v in CLASSES_need.items()}
    # print(desired_class_name_to_value)
    # 已有数据集的类别名称到类别值的映射
    class_value_mapping = {}
    # 遍历已有数据集的类别名称到类别值的映射
    has_label = False
    need_convert = False

    # 增加判断，是否需要转换
    for class_info in classes_pri:
        class_name = class_info["classname_en"]
        class_value = class_info["class_value"]
        # if class_name.lower() == 'background':
        #     has_background = True
        # 如果已有数据集的类别名称在你想要的类别名称到类别值的映射中
        if class_name in desired_class_name_to_value:
            if class_value != desired_class_name_to_value[class_name]:
                need_convert = True
            # 将已有数据集的类别值映射到你想要的类别值
            class_value_mapping[class_value] = desired_class_name_to_value[class_name]
            has_label = True
    # logger.info(class_value_mapping)
    # print(has_background)
    if not need_convert:
        class_value_mapping = None
    return class_value_mapping, has_label


def match_images_and_labels(image_list, label_list):
    # 获取图像和标签的文件名（不包含扩展名）
    logger.info("==> begin check the matching of image and label")
    image_names = [os.path.splitext(os.path.basename(image))[0] for image in image_list]
    label_names = [os.path.splitext(os.path.basename(label))[0] for label in label_list]

    # 找到匹配的图像和标签
    matched_images = []
    unmatched_count = 0
    for image_name, image in zip(image_names, image_list):
        if image_name in label_names:
            label =label_list[label_names.index(image_name)]
            matched_images.append([image, label])
            label_names.remove(image_name)
            label_list.remove(label)
            # matched_labels.append(label_list[label_names.index(image_name)])
        else:
            unmatched_count += 1
    logger.info(f"Down! {unmatched_count} images have not label")
    return matched_images



if __name__ == "__main__":
    args = args_parser()
    from datetime import datetime

    # 获取当前日期和时间
    now = datetime.now()

    # 格式化日期和时间
    datetime_begin = now.strftime("%Y-%m-%d")

    logger.info(args.dataset_path)

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = False
    # if args.providers.lower() == 'cuda':
    #     Cuda = True
    # ----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed .launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # -----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # -----------------------------------------------------#
    # CLASS_ALL = ast.literal_eval(args.classes)
    # CLASSES_need_label = args.categorys

    CLASS_ALL = args.classes.copy()
    for class_label in args.classes:
        if class_label.lower() == "background":
            CLASS_ALL.remove(class_label)

    CLASSES_need = {i + 1: class_name for i, class_name in enumerate(CLASS_ALL)}
    logger.info(CLASSES_need)  #
    num_classes = len(CLASSES_need) + (0 if CLASSES_need.get("background") else 1)
    # -----------------------------------------------------#
    #   主干网络选择
    #   vgg
    #   resnet50
    # -----------------------------------------------------#
    backbone = args.backbone
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # 先判断是否进行迁移学习，如果进行，则从迁移学习的路径读取
    # 否，则判断是否使用预训练的backbone，如果使用，则读取backbone存储路径
    # 最后加载权重文件
    all_args = {'model': 'unet', **vars(args)}
    # all_args['classes'] = ast.literal_eval(all_args['classes'])
    os.makedirs(args.weight_save_dir, exist_ok=True)
    logger.info(all_args)
    with open(os.path.join(args.weight_save_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(all_args, f, indent=4)

    model_path = ''
    classes_match_flag = False
    if args.transfer_path != '':
        model_path = os.path.join(args.transfer_path, "best_epoch_weights.pth")  # 需要修改一下，兼容backbone的
        model_meta_param_path = os.path.join(args.transfer_path, 'model_param.json')
        with open(model_meta_param_path, 'r', encoding='utf-8') as ft:
            transfer_model_param = json.load(ft)
        transfer_model_category = transfer_model_param['category']
        if transfer_model_category == CLASS_ALL:
            classes_match_flag = True
    elif args.pretrained.lower() == 'true':
        if args.backbone == 'vgg':
            model_path = os.path.join(os.path.dirname(__file__), 'model_data/unet_vgg_voc.pth')
        elif args.backbone == 'resnet50':
            model_path = os.path.join(os.path.dirname(__file__), 'model_data/unet_resnet50_voc.pth')
    # -----------------------------------------------------#
    #   input_shape     输入图片的大小，32的倍数
    # -----------------------------------------------------#
    input_shape = [512, 512]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #   
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合语义分割，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在120-300之间调整。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       由于resnet50中有BatchNormalization层
    #       当主干为resnet50的时候batch_size不可为1
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = 0  # args.init_epoch
    Freeze_Epoch = args.training_epoch // 3  # 冻结训练20% # args.freeze_epoch
    Freeze_batch_size = args.batch_size
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = args.training_epoch
    Unfreeze_batch_size = args.batch_size
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    if Freeze_Epoch > 0:
        Freeze_Train = True
    else:
        Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = args.init_lr
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = args.training_epoch
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    # save_dir = args.save_dir
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = args.eval_epoch

    # ------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ------------------------------------------------------------------#
    dice_loss = True if args.dice_loss.lower() == 'true' else False
    # ------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡, 默认为CEloss
    # ------------------------------------------------------------------#
    focal_loss = True if args.focal_loss.lower() == 'true' else False
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 0

    seed_everything(seed)
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            logger.info(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            logger.info("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if (torch.cuda.is_available() and Cuda) else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone, in_channels=args.in_channels).train()
    if args.in_channels ==3:
        from utils.dataloader import UnetDataset, unet_dataset_collate
    else:
        from utils.tif import build_costum_tif_loader as UnetDataset
        from utils.tif import unet_dataset_collate
    if not pretrained:
        weights_init(model)

    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            logger.info('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        # 在这里修改，加载最后一层分类的参数

        for k, v in pretrained_dict.items():
            if k in ["final.weight", "final.bias"]:
                if not classes_match_flag:  # 判断一下迁移学习的分类类别与现在的是否完全一致,
                    no_load_key.append(k)
                    continue
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            logger.info("Successful Load Key:")
            logger.info(str(load_key)[:500])
            logger.info("Successful Load Key Num:" + str(len(load_key)))
            logger.info("Fail To Load Key:")
            logger.info(str(no_load_key)[:500])
            logger.info("Fail To Load Key num:" + str(len(no_load_key)))
            logger.info("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    # if local_rank == 0:
    #     time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    #     log_dir = os.path.join(save_dir, "loss_" + str(time_str))  # 是否需要修改，日志存放路径
    #     loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    # else:
    #     loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        logger.info("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 如果要实现根据输入比例来划分训练验证集，需要对此进行处理，同时还需处理，让他兼容其他类型的数据集？
        # 根据比例划分训练验证数据：

        # ------------------------------#
        #   数据集路径
        # ------------------------------#
        # 获取数据集的名称，从而获得对应的数据集路径等元信息
        VOCdevkit_paths = args.dataset_path  #
        logger.info(VOCdevkit_paths)
        train_dataset = None
        val_dataset = None
        num_train = 0
        num_val = 0
        # 判断读取的数据集是单个还是多个
        split_rate = 0.8
        for VOCdevkit_path in VOCdevkit_paths:
            # 增加数据集metajson的读取，获得数据集信息
            with open(os.path.join(VOCdevkit_path, 'meta.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 获取类别数
            classes_pri = metadata['classes']
            image_list = os.listdir(os.path.join(VOCdevkit_path, "jpg"))
            label_list = os.listdir(os.path.join(VOCdevkit_path, "anno"))
            data_names = match_images_and_labels(image_list, label_list)
            del image_list, label_list
            # data_names = [i[:-4] for i in data_names]
            random.shuffle(data_names)
            split_index = int(len(data_names) * split_rate)
            train_lines = data_names[:split_index]
            val_lines = data_names[split_index:]
            num_train += len(train_lines)
            num_val += len(val_lines)
            convert_map, has_label = convert_label(CLASSES_need, classes_pri)

            if not has_label:
                logger.warning("当前数据集不包含要预测目标的有效标签")
                continue
            # num_class = len(classes_pri) + (0 if has_background else 1)  # 1表示增加背景类
            if train_dataset is None:
                # train_dataset = build_costum_tif_loader(train_lines, input_shape, num_classes, VOCdevkit_path,
                #                         convert_map, args.in_channels)
                train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path,
                                            convert_map, args.in_channels)  # 传入数据增强的参数
                val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path,
                                          convert_map, args.in_channels)
            else:
                train_dataset += UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path,
                                             convert_map, args.in_channels)  # 传入数据增强的参数
                val_dataset += UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path,
                                           convert_map, args.in_channels)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if local_rank == 0:
            show_config(
                num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
                Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
                Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
                Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
                lr_decay_type=lr_decay_type, log_save_dir=args.log_save_dir, weight_save_dir=args.weight_save_dir,
                num_workers=num_workers, num_train=num_train,
                num_val=num_val, devive=device
            )

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        # if local_rank == 0:
        #     eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
        #                                  eval_flag=eval_flag, period=eval_period)
        # else:
        #     eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        step = 0
        avg_loss = 100.0
        iou = 0
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则当epoch>freeze_epoch时解冻,并设置参数,重新设置训练参数batch size,lr等
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                if train_dataset is None:
                    sys.exit()
                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            step, avg_loss, iou = fit_one_epoch(model_train, model, args, step, optimizer, epoch,
                                                epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                                                dice_loss, focal_loss,
                                                cls_weights, num_classes, fp16, scaler, local_rank, avg_loss, iou)

            if distributed:
                dist.barrier()

        # # 获取当前日期和时间
        # now = datetime.now()
        # # 格式化日期和时间
        # datetime_end = now.strftime("%Y-%m-%d %H:%M:%S")

            model_param_dict = {"modelName": "UNet-01",
                                "baseModel": "Unet",
                                "backbone": args.backbone,
                                "modelType": "landcover-classfication",
                                "modelVersion": "1.0.0",
                                "modelDescription": "模型说明",
                                "category": list(CLASSES_need.values()),
                                "Accuray": round(iou, 2),
                                "author": "...",
                                "create-time": datetime_begin,
                                # "end-time": datetime_end
                                }
            with open(os.path.join(args.weight_save_dir, 'model_param.json'), 'w', encoding='utf-8') as ff:
                json.dump(model_param_dict, ff, indent=4, ensure_ascii=False)
        # if local_rank == 0:
        #     loss_history.writer.close()
