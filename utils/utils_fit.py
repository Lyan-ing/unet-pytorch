import json
import os

import numpy as np
import torch
import torch.distributed as dist
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.iou import AverageMeter, intersectionAndUnion

def fit_one_epoch(model_train, model, args, step, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                  local_rank=0, avg_loss=0.0, iou = 0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    # model_train.eval()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # for iteration, batch in enumerate(gen_val):
    #     if iteration >= epoch_step_val:
    #         break
    #     imgs, pngs, labels = batch
    #     with torch.no_grad():
    #         weights = torch.from_numpy(cls_weights)
    #         if cuda:
    #             imgs = imgs.cuda(local_rank)
    #             pngs = pngs.cuda(local_rank)
    #             labels = labels.cuda(local_rank)
    #             weights = weights.cuda(local_rank)
    #
    #         # ----------------------#
    #         #   前向传播
    #         # ----------------------#
    #         outputs = model_train(imgs)
    #         pred = outputs.argmax(dim=1)
    #         intersection, union, target = \
    #             intersectionAndUnion(pred.cpu().numpy(), pngs.cpu().numpy(), num_classes, 255)
    #
    #         intersection_meter.update(intersection)
    #         union_meter.update(union)
    #         # ----------------------#
    #         #   损失计算
    #         # ----------------------#
    #         if focal_loss:
    #             loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
    #         else:
    #             loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
    #
    #         if dice_loss:
    #             main_dice = Dice_loss(outputs, labels)
    #             loss = loss + main_dice
    #         # -------------------------------#
    #         #   计算f_score
    #         # -------------------------------#
    #         _f_score = f_score(outputs, labels)
    #
    #         val_loss += loss.item()
    #         val_f_score += _f_score.item()
    #     avg_val_loss = val_loss / (iteration + 1)
    #
    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    # print(iou_class)
    # mIOU = np.mean(iou_class)
    #

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    with open(args.log_save_dir, 'a') as f:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

            optimizer.zero_grad()
            if not fp16:
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

                loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    # ----------------------#
                    #   前向传播
                    # ----------------------#
                    outputs = model_train(imgs)
                    # ----------------------#
                    #   损失计算
                    # ----------------------#
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        loss = loss + main_dice

                    with torch.no_grad():
                        # -------------------------------#
                        #   计算f_score
                        # -------------------------------#
                        _f_score = f_score(outputs, labels)

                # ----------------------#
                #   反向传播
                # ----------------------#
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            log_dict = {"flag": "train", "step": step, "epoch": epoch, "loss": loss.item(), "lr": get_lr(optimizer)}
            f.write('\n')  # 添加换行符
            json.dump(log_dict, f)
            step += 1

            if local_rank == 0:
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'f_score': total_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            pred = outputs.argmax(dim=1)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), pngs.cpu().numpy(), num_classes, 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()
        avg_val_loss = val_loss / (iteration + 1)

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': avg_val_loss,
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    print(iou_class)
    mIOU = np.mean(iou_class)

    val_save_dict = {"flag": "val", "step": epoch, "epoch": epoch, "avg_val_loss": avg_val_loss, "mIOU": mIOU}
    print(val_save_dict)
    with open(args.log_save_dir, 'a') as fv:
        fv.write('\n')  # 添加换行符
        json.dump(val_save_dict, fv)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        if avg_val_loss < avg_loss:
            avg_loss = avg_val_loss
            iou = mIOU
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(args.weight_save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(args.weight_save_dir, "last_epoch_weights.pth"))
        # with open('mo')
    return step, avg_loss, iou


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print()
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
