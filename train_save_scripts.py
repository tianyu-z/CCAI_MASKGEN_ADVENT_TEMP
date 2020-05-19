import warnings
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from collections import deque
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import (
    adjust_learning_rate,
    adjust_learning_rate_discriminator,
    loss_calc,
    bce_loss,
    prob_2_entropy,
    per_class_iu,
    fast_hist,
)
from advent.utils.tools import (
    print_losses,
    tesnorDict2numDict,
    write_images
)
from time import time
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def train_preview(model, source_loader, target_loader, cfg, comet_exp):
    # UDA TRAINING
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    times = deque([0], maxlen=100)
    model_times = deque([0], maxlen=100)
    
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)

    cur_best_miou = -1
    cur_best_model = ''

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        times.append(time())
        comet_exp.log_metric("i_iter", i_iter)

        comet_exp.log_metric("target_epoch", i_iter/len(target_loader))
        comet_exp.log_metric("source_epoch", i_iter/len(source_loader))
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        try:
            _, batch_and_path = source_loader_iter.__next__()
        except StopIteration:
            source_loader_iter = enumerate(source_loader)
            _, batch_and_path = source_loader_iter.__next__()

        images_source, labels = batch_and_path['data']['x'], batch_and_path['data']['m']
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training to fool the discriminator
        try:
            _, batch = target_loader_iter.__next__()
        except StopIteration:
            target_loader_iter = enumerate(target_loader)
            _, batch = target_loader_iter.__next__()

        images = batch['data']['x']
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()
        
        model_times.append(time() - times[-1])
        mod_times = np.mean(model_times)
        comet_exp.log_metric("model_time", mod_times)
        
        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)
        current_losses_numDict = tesnorDict2numDict(current_losses)
        comet_exp.log_metrics(current_losses_numDict)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break

        if i_iter % cfg.TRAIN.SAVE_IMAGE_PRED == 0 and i_iter != 0 or i_iter == cfg.TRAIN.EARLY_STOP:
            print("Inferring test images in iteration {}...".format(i_iter))
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            image, label = batch['data']['x'][0], batch['data']['m'][0]
            image = image[None, :, :, :]

            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            with torch.no_grad():
                pred_main = model(image.cuda(device))[1]
                output = interp(pred_main).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)
            label0 = label.numpy()[0]
            hist += fast_hist(label0.flatten(), output.flatten(), cfg.NUM_CLASSES)
            output = torch.tensor(output, dtype=torch.float32)
            output = output[None, :, :]
            output_RGB = output.repeat(3, 1, 1)
          
            if i_iter % 100 == 0:
                print('{:d} / {:d}: {:0.2f}'.format(
                    i_iter % len(target_loader), len(target_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
            if cur_best_miou < computed_miou:
                cur_best_miou = computed_miou
                cur_best_model = f'model_{i_iter}.pth'
            print('\tCurrent mIoU:', computed_miou)
            print('\tCurrent best model:', cur_best_model)
            print('\tCurrent best mIoU:', cur_best_miou)
            mious = {'Current mIoU': computed_miou,
                          'Current best model': cur_best_model,
                          'Current best mIoU': cur_best_miou}
            comet_exp.log_metrics(mious)
            image = image[0] # change size from [1,x,y,z] to [x,y,z]
            save_images = []

            save_images.append(image)
            # Overlay mask:

            save_mask = (
                image
                - (image * label.repeat(3, 1, 1))
                + label.repeat(3, 1, 1)
            )

            save_fake_mask = (
                image
                - (image * output_RGB)
                + output_RGB
            )
            save_images.append(save_mask)
            save_images.append(save_fake_mask)
            save_images.append(label.repeat(3, 1, 1))
            save_images.append(output_RGB)

            write_images(
                save_images,
                i_iter,
                comet_exp=comet_exp,
                store_im=cfg.TEST.store_images
            )
