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
from dqn_agent import ObjDetEnv
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

    # Initiate DQN agent.
    init_item = next(iter(loader))
    print("Begin building DQN agent ......")
    env = ObjDetEnv(init_item[1],
                    init_item[2][:, :, :, :6],
                    batch_size,
                    dataset_des,
                    init_dim_cluster)
    print("Complete building DQN agent")

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

        env.reset(img, raw_targets[:, :, :, :6])
        input_img = env.get_img()

        with torch.no_grad():
            pred_conf_cls, _ = model(input_img, env.get_bbox())
            pred_conf_cls = pred_conf_cls.permute(0, 2, 3, 1)
            pred_conf = torch.sigmoid(pred_conf_cls[:, :, :, 0])
            pred_cls = torch.sigmoid(pred_conf_cls[:, :, :, 1:])
            obj_mask = pred_conf > conf_thres
            obj_mask = obj_mask.byte().to('cuda')

        while True:
            current_bbox = env.get_bbox()
            with torch.no_grad():
                _, acts_value = model(input_img, current_bbox)
            _, acts = torch.max(acts_value, 1)

            acts[env.done] = 0
            # print(acts[env.object_mask])
            done = env.eval_step(acts)

            batch_step_counter += 1

            # if torch.sum(done) >= (13 * 13 * batch_size) or batch_step_counter >= (env.max_step + 1):
            if sum(done[obj_mask]) == len(done[obj_mask]) or batch_step_counter >= (env.max_step + 1):
                print("object numbers: {} -- batch_step: {}".format(torch.sum(obj_mask), env.steps))
                break

        if diagnosis_code == 0:
            pred_bbox = env.get_bbox()
            pred_bbox = pred_bbox.permute(0, 2, 3, 1)  # torch.Size([B, 13, 13, 4])
            # pred_conf_cls = pred_conf_cls.permute(0, 2, 3, 1)
            # pred_conf = torch.sigmoid(pred_conf_cls[:, :, :, 0])
            # pred_cls = torch.sigmoid(pred_conf_cls[:, :, :, 1:])
        # Error diagnosis, i.e., feeding the ground-truth
        if diagnosis_code == 1:
            # localization ground-truth
            pred_bbox = raw_targets[:, :, :, 2:6]
            # pred_conf_cls = pred_conf_cls.permute(0, 2, 3, 1)
            # pred_conf = torch.sigmoid(pred_conf_cls[:, :, :, 0])
            # pred_cls = torch.sigmoid(pred_conf_cls[:, :, :, 1:])
        if diagnosis_code == 2:
            # classification ground-truth
            # pred_bbox = env.init_bbox
            pred_bbox = env.get_bbox()
            pred_bbox = pred_bbox.permute(0, 2, 3, 1)  # torch.Size([B, 13, 13, 4])
            pred_conf = raw_targets[:, :, :, 0]
            pred_cls = raw_targets[:, :, :, 6:]
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

        # # For debug
        # if i_batch > 3:
        #     break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def main():
    # Hyperparameters parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default='2012', help="used to select training set")
    parser.add_argument("--set", type=str, default='val', help="used to select training set")
    # parser.add_argument("--epochs", type=int, default=101, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_large.cfg", help="path to model definition file")
    parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_mini.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/net/dqnyolo_tiny.cfg", help="path to model definition file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--gamma", type=float, default=0.9, help="discount factor in DQN")
    # parser.add_argument("--init_epsilon", type=float, default=0, help="init epsilon greedy policy")
    # parser.add_argument("--incre_epsilon", type=float, default=0.016, help="increment epsilon greedy policy")
    # parser.add_argument("--opt_lr", type=float, default=1e-3, help="learning rate for optimizer")
    # parser.add_argument("--target_replace_iter", type=int, default=3, help="interval to update target network")
    parser.add_argument("--use_gpu", default=True, help="use GPU to accelerate training")
    parser.add_argument("--init_dim_cluster", default=True, help="whether adopt initial bboxes dimension cluster.")
    # parser.add_argument("--parallel_step", default=True, help="whether parallel transformation in dqn_agent")
    # parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    # parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on training set")
    opt = parser.parse_args()
    print(opt)

    if opt.use_gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise RuntimeError("Current Torch doesn't have GPU support.")
    else:
        device = torch.device('cpu')

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    print("Begin loading model in {}".format("./logs/trained/master-v2/mini/model_params_360.ckpt"))
    model.load_state_dict(torch.load("./logs/trained/master-v2/mini/model_params_360.ckpt"))
    print("Complete loading model")
    # print(model)
    # summary(eval_model, [(3, 416, 416), (4, 13, 13)])

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        [opt.year, opt.set],
        [0.5, 0.5, 0.5],
        opt.batch_size,
        opt.init_dim_cluster,
        diagnosis_code=2
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    print("evaluation_metrics", evaluation_metrics)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, val2labels(c), "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")


if __name__ == '__main__':
    # dataset = load_voc_test("./data/VOC/", '2012')
    # trans_voc(dataset)
    main()
