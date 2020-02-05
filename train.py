"""
Train the model.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import argparse
import numpy as np
import pickle
import time
from torchsummary import summary
from warmup_scheduler import GradualWarmupScheduler
from terminaltables import AsciiTable

from utils.datasets import trans_voc, ListDataset, val2labels
from utils.utils import weights_init_normal, exist_or_create_folder, plot_epoch_info
from model import Darknet
from dqn_agent import ObjDetEnv
from detect import evaluate


def main():
    # Hyperparameters parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default='2012', help="used to select training set")
    parser.add_argument("--set", type=str, default='train', help="used to select training set")
    parser.add_argument("--epochs", type=int, default=201, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/net/resnet.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_large.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_mini.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_tiny.cfg", help="path to model definition file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--opt_lr", type=float, default=1e-5, help="learning rate for optimizer")
    parser.add_argument("--use_gpu", default=True, help="use GPU to accelerate training")
    parser.add_argument("--shuffle_train", default=True, help="shuffle the training dataset")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    # parser.add_argument("--pretrained_weights", type=str, default="data/backbone/darknet53.conv.74", help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_weights", type=str, default="logs/model/model_params_200.ckpt", help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_weights", default=False, help="if specified starts from checkpoint model")
    opt = parser.parse_args()
    print(opt)

    if opt.use_gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise RuntimeError("Current Torch doesn't have GPU support.")
    else:
        device = torch.device('cpu')

    logger = SummaryWriter(exist_or_create_folder("./logs/tb/"))

    # Initiate model
    # model = Darknet(opt.model_def).to(device)
    eval_model = Darknet(opt.model_def).to(device)
    if opt.pretrained_weights:
        print("Initialize model with pretrained_model")
        if opt.pretrained_weights.endswith(".ckpt"):
            eval_model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            eval_model.load_darknet_weights(opt.pretrained_weights)
    else:
        print("Initialize model randomly")
        eval_model.apply(weights_init_normal)
    # eval_model.load_state_dict(torch.load("./logs/saved_exp/master-v2/model_params_80.ckpt"))
    # print(model)
    summary(eval_model, (3, 416, 416))

    learn_epoch_counter = 0  # (total numbers)
    learn_batch_counter = 0  # for logger update (total numbers)
    # learn_step_counter = 0  # for logger update (total numbers)
    batch_size = opt.batch_size

    # Get dataloader
    print("Begin loading train dataset ......")
    t_load_data = time.time()
    dataset = torchvision.datasets.VOCDetection(root='data/VOC/',
                                                year=opt.year,
                                                image_set=opt.set,
                                                transforms=None,
                                                download=True)
    dataset_dict = trans_voc(dataset)
    dataset = ListDataset(dataset_dict)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle_train,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    print("Complete loading train dataset in {} s".format(time.time() - t_load_data))

    optimizer = torch.optim.Adam(eval_model.parameters(), lr=opt.opt_lr)
    # Warmup and learning rate decay
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    # 5 epoch warmup, lr from 1e-5 to 1e-4, after that schedule as after_scheduler
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10,
                                              after_scheduler=scheduler_cosine)

    start_time = time.time()
    epoch_info = {}

    for i_epoch in range(opt.epochs):
        eval_model.train()

        for i_batch, (_, imgs, raw_targets, transform_params, tar_boxes) in enumerate(loader):
            print("\n++++++++++ i_epoch-i_batch {}-{} ++++++++++".format(i_epoch, i_batch))
            batch_step_counter = 0
            batch_loss = []

            if len(imgs) != opt.batch_size:
                print("Current batch size is smaller than opt.batch_size!")
                continue

            # print("imgs.size()")
            # print(imgs.size())
            # print("raw_targets")
            # print(raw_targets.size())
            # print(raw_targets)
            # imgs: [3, 416, 416]
            # imgs = Variable(imgs.to(device))
            # raw_targets = Variable(raw_targets.to(device), requires_grad=False)
            imgs = imgs.to(device)
            raw_targets = raw_targets.to(device)
            tar_boxes = tar_boxes.to(device)

            input_img = imgs

            if i_epoch == 0 and i_batch == 0:
                logger.add_graph(eval_model, input_img)

            # print(raw_targets)
            # print(raw_targets.size())
            # print(raw_targets[:, :, :, 6:].size())
            # print(raw_targets[:, :, :, 0].unsqueeze(3).size())
            cls_targets = torch.cat((raw_targets[:, :, :, 0].unsqueeze(3), raw_targets[:, :, :, 6:]), 3)
            # print(cls_targets.size())

            loss = eval_model(input_img, cls_targets)

            batch_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            # Accumulates gradient before each step
            # if learn_batch_counter % opt.gradient_accumulations:
            optimizer.step()

            batch_step_counter += 1
            # learn_step_counter += 1

            # Post-processing after one batch.
            learn_batch_counter += 1

            print("Ep-bt: {}-{} | Loss: {}".format(i_epoch, i_batch, loss.item()))

            batch_loss_mean = sum(batch_loss) / len(batch_loss)
            logger.add_scalar('loss/loss', batch_loss_mean, learn_batch_counter)

        # Post-processing after one epoch.
        learn_epoch_counter += 1

        if learn_epoch_counter % opt.checkpoint_interval == 0:
            print("Saving model in epoch {}".format(learn_epoch_counter))
            torch.save(eval_model.state_dict(),
                       exist_or_create_folder("./logs/model/model_params_{}.ckpt".format(learn_epoch_counter)))

        # Evaluate the model on the validation set
        if learn_epoch_counter % opt.evaluation_interval == 0:
            pass

        # Evaluate the model on the training set
        if learn_epoch_counter % opt.evaluation_interval == 0:
            pass

        # Warmup and lr decay
        scheduler_warmup.step()

        # Free GPU memory
        torch.cuda.empty_cache()

    total_train_time = time.time() - start_time
    print("Training complete in {} hours".format(total_train_time / 3600))


if __name__ == '__main__':
    main()
