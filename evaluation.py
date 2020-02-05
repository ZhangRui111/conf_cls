"""
Model evaluation and do detection.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import argparse
import numpy as np
import time

from utils.datasets import trans_voc, ListDataset, plot_detection_result, val2labels, load_voc_test
from utils.utils import non_max_suppression, get_batch_statistics, ap_per_class
from model import Darknet
from PIL import Image
from terminaltables import AsciiTable


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(model, dataset_des, thres, batch_size, init_dim_cluster, diagnosis_code=0):
    model.eval()
    iou_thres, conf_thres, nms_thres = thres

    # Get dataloader
    print("Begin loading validation dataset ......")
    t_load_data = time.time()
    dataset = torchvision.datasets.VOCDetection(root='data/VOC/',
                                                year=dataset_des[0],
                                                image_set=dataset_des[1],
                                                transforms=None,
                                                download=True)
    dataset_dict = trans_voc(dataset)
    dataset = ListDataset(dataset_dict)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    print("Complete loading validation dataset in {} s".format(time.time() - t_load_data))

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for i_batch, (img_ind, img, raw_targets, transform_params, tar_boxes) in enumerate(loader):
        print("\n++++++++++ i_batch (val) {} ++++++++++".format(i_batch))
        batch_step_counter = 0

        # Extract labels: raw_targets -- [t_conf, t_cls, x, y, w, h, one_hot_t_cls(20)]
        labels += tar_boxes[:, 1].tolist()

        if len(img) != batch_size:
            print("Current batch size is smaller than opt.batch_size!")
            continue

        img = img.to('cuda')
        raw_targets = raw_targets.to('cuda')
        tar_boxes = tar_boxes.to('cuda')

        input_img = img

        with torch.no_grad():
            pred_conf_cls = model(input_img)
            pred_conf_cls = pred_conf_cls.permute(0, 2, 3, 1)
            pred_conf = torch.sigmoid(pred_conf_cls[:, :, :, 0])
            pred_cls = torch.sigmoid(pred_conf_cls[:, :, :, 1:])
            obj_mask = pred_conf > conf_thres
            obj_mask = obj_mask.byte().to('cuda')

        if diagnosis_code == 0:
            pass
        if diagnosis_code == 1:
            # localization ground-truth
            pred_bbox = raw_targets[:, :, :, 2:6]
            # pred_conf_cls = pred_conf_cls.permute(0, 2, 3, 1)
            # pred_conf = torch.sigmoid(pred_conf_cls[:, :, :, 0])
            # pred_cls = torch.sigmoid(pred_conf_cls[:, :, :, 1:])
        if diagnosis_code == 2:
            # classification ground-truth
            pass
        if diagnosis_code == 3:
            # full ground-truth
            pred_bbox = raw_targets[:, :, :, 2:6]
            pred_conf = raw_targets[:, :, :, 0]
            pred_cls = raw_targets[:, :, :, 6:]

        # if i_batch < 20:
        #     im = Image.open("data/VOC/VOCdevkit/VOC2007/JPEGImages/{}".format(img_ind[0])).convert('RGB')
        #     plot_detection_result(im, img_ind, pred_bbox, pred_conf, pred_cls, transform_params, iou=0.9)

        pred_outputs = torch.cat((pred_bbox, pred_conf.unsqueeze(3), pred_cls), dim=3)
        b, w, h, d = pred_outputs.shape
        pred_outputs = pred_outputs.view(b, w * h, d)

        outputs = non_max_suppression(pred_outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, tar_boxes, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
