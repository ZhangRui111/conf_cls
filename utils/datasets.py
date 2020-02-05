"""
Dataset related operations.
xywh format: (center_x, center_y, width, height)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import transforms

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import xml.etree.ElementTree as ET
# import sys
# import time

from utils.utils import exist_or_create_folder

# sys.path.append("utils/")


def get_additional_channel(img_path, mode):
    """
    Build additional channel for the input image.
    :param img_path:
    :param mode:
    :return: ndarray
    """
    raw_img = cv2.imread(img_path, flags=0)
    if mode == 'canny':
        img = cv2.Canny(raw_img, 100, 200)
    elif mode == 'laplacian':
        img = cv2.Laplacian(raw_img, cv2.CV_64F)
    else:
        raise ValueError("Invalid mode {}".format(mode))
    return img


def horizontal_flip(images, targets):
    """
    horisontal flip.
    :param images: (c, h, w)
    :param targets: (_, cls, x, y, w, h)
    :return:
    """
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def vertical_flip(images, targets):
    """
    vertical flip.
    :param images: (c, h, w)
    :param targets: (_, cls, x, y, w, h)
    :return:
    """
    images = torch.flip(images, [-2])
    targets[:, 3] = 1 - targets[:, 2]
    return images, targets


def load_classes(path):
    """ Loads class labels at 'path'. """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def xywh2xyxy(bbox):
    """
    [x_center, y_center, width, height] --> [x_min, y_min, x_max, y_max]
    :param bbox:
    :return:
    """
    return [bbox[0] - bbox[2]/2,
            bbox[1] - bbox[3]/2,
            bbox[0] + bbox[2]/2,
            bbox[1] + bbox[3]/2]


def xyxy2xywh(bbox):
    """
    [x_min, y_min, x_max, y_max] --> [x_center, y_center, width, height]
    :param bbox:
    :return:
    """
    return [(bbox[0] + bbox[2])/2,
            (bbox[1] + bbox[3])/2,
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]]


def plot_detection_result(img, img_ind, pred_bbox, pred_conf, pred_cls, transform_params, iou):
    """
    Plot detection result by bboxes.
    :param img: PIL.image
    :param pred_bbox:
    :param pred_conf:
    :param pred_cls:
    :param transform_params:
    :param iou:
    :return:
    """
    assert pred_bbox.shape[0] == 1, "detection with GUI requests batch_size==1"
    pred_conf, pred_cls, pred_bbox = pred_conf.squeeze(0), pred_cls.squeeze(0), pred_bbox.squeeze(0)
    # img.show()
    img.save("data/detection_samples/{}".format(img_ind[0]))
    obj_mask = (pred_conf > iou).byte().to("cuda")
    # print(obj_mask)
    _, pred_cls = torch.max(pred_cls, dim=2)
    obj_cls = pred_cls[obj_mask]
    obj_bbox = pred_bbox[obj_mask]
    # Transform obj_bbox to the raw image (i.e., revert rescale & padding)
    transform_params = transform_params.squeeze(0)
    obj_bbox *= (transform_params[-1] / 416)
    paddinf_transform = torch.from_numpy(
        np.asarray([transform_params[0],
                    transform_params[2],
                    transform_params[1],
                    transform_params[3]])).to("cuda").float()
    for ind in range(obj_bbox.shape[0]):
        bbox = obj_bbox[ind, :]
        bbox -= paddinf_transform
        bbox = xywh2xyxy(bbox)
        bbox_draw = ImageDraw.Draw(img)
        bbox_draw.rectangle(bbox, outline="red")
        label_draw = ImageDraw.Draw(img)
        label_draw.text((bbox[-2], bbox[-3]), val2labels(obj_cls[ind].item()))
    # img.show("Image with bbox")
    img.save("data/detection_samples/det_{}".format(img_ind[0]))


def plot_bbox(img, img_ind, labels, bboxes, colors):
    """
    Plot bboxes and labels in an image.
    :param img: PIL.Image.Image
    :param img_ind: image index
    :param labels: a list consisting of all bboxes' labels.
    :param bboxes: a list consisting of all bboxes.  [x_center, y_center, width, height]
    :param colors: a list consisting of all bboxes' colors.
    :return:
    """
    for label, bbox, color in zip(labels, bboxes, colors):
        bbox = xywh2xyxy(bbox)
        bbox_draw = ImageDraw.Draw(img)
        bbox_draw.rectangle(bbox, outline=color)
        label_draw = ImageDraw.Draw(img)
        label_draw.text((bbox[-2], bbox[-3]), label)
    # img.save("../data/samples/{}".format(img_ind), "png")
    img.show("Image with bbox")


def show_sample_img(img, annota):
    """
    Show a sample image with bboxes and labels.
    :param img:
    :param annota:
    :return:
    """
    labels, bboxes, colors = [], [], []
    img_ind = annota['annotation']['filename']
    objs = annota['annotation']['object']
    n_objs = len(objs)

    if type(objs) == list:
        n_objs = len(objs)
        for i in range(n_objs):
            label = objs[i]['name']
            bbox = objs[i]['bndbox'].values()  # x_min, y_min, x_max, y_max
            bbox = [item for item in map(int, bbox)]
            bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
            labels.append(label)
            bboxes.append(bbox)
            colors.append('red')
    elif type(objs) == dict:
        n_objs = 1
        label = objs['name']
        bbox = objs['bndbox'].values()  # x_min, y_min, x_max, y_max
        bbox = [item for item in map(int, bbox)]
        bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
        labels.append(label)
        bboxes.append(bbox)
        colors.append('red')

    img.show(title="Raw image")
    plot_bbox(img, img_ind, labels, bboxes, colors)


def padding_square_np(img, pad_value=0):
    """
    Padding input image (numpy format) to square.
    :param img: input image (PIL.image).
    :param size: resized size.
    :param pad_value:
    :return:
    """
    # img.show()
    img_np = np.asarray(img)
    h, w, c = img_np.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    img_np = np.pad(img_np, pad, mode="constant", constant_values=pad_value)

    img_pic = Image.fromarray(img_np)
    # Varification on padding results
    # img_pic.show()

    return img_pic, pad


def pad_to_square(img, pad_value):
    """
    Padding input image (torch format) to square.
    :param img: input image (PIL.image).
    :param size: resized size.
    :param pad_value:
    :return:
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = [0, 0, pad1, pad2] if h <= w else [pad1, pad2, 0, 0]
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    # Varification on padding results
    # img.show()

    return img, pad


def square_resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def labels2val(label):
    trans_dict = {
        'person': 0,
        'bird': 1,
        'cat': 2,
        'cow': 3,
        'dog': 4,
        'horse': 5,
        'sheep': 6,
        'aeroplane': 7,
        'bicycle': 8,
        'boat': 9,
        'bus': 10,
        'car': 11,
        'motorbike': 12,
        'train': 13,
        'bottle': 14,
        'chair': 15,
        'diningtable': 16,
        'pottedplant': 17,
        'sofa': 18,
        'tvmonitor': 19,
    }
    return int(trans_dict[label])


def val2labels(val):
    trans_dict = {
        0: 'person',
        1: 'bird',
        2: 'cat',
        3: 'cow',
        4: 'dog',
        5: 'horse',
        6: 'sheep',
        7: 'aeroplane',
        8: 'bicycle',
        9: 'boat',
        10: 'bus',
        11: 'car',
        12: 'motorbike',
        13: 'train',
        14: 'bottle',
        15: 'chair',
        16: 'diningtable',
        17: 'pottedplant',
        18: 'sofa',
        19: 'tvmonitor',
    }
    return trans_dict[val]


def read_content(xml_file: str):
    """
    Modified based on: https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
    :param xml_file:
    :return:
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotation = {'filename': root.find('filename').text}

    size = root.find('size')
    annot_size = {'width': int(size.find('width').text),
                  'height': int(size.find('height').text),
                  'depth': int(size.find('depth').text)}
    annotation['size'] = annot_size

    obj_list = []
    for obj in root.iter('object'):
        annot_obj = {'name': obj.find("name").text}
        box = obj.find("bndbox")
        ymin = int(box.find("ymin").text)
        xmin = int(box.find("xmin").text)
        ymax = int(box.find("ymax").text)
        xmax = int(box.find("xmax").text)
        annot_obj['bndbox'] = {'xmin': xmin,
                               'xmax': xmax,
                               'ymin': ymin,
                               'ymax': ymax}
        obj_list.append(annot_obj)

    if len(obj_list) == 1:
        annotation['object'] = obj_list[0]
    else:
        annotation['object'] = obj_list

    return {'annotation': annotation}


def load_voc_test(root, year):
    dataset = []

    root_path = "{}VOC{}_test/".format(root, year)
    image_path = "{}JPEGImages/".format(root_path)
    annot_path = "{}Annotations/".format(root_path)
    ind_path = "{}ImageSets/Main/test.txt".format(root_path)

    with open(ind_path, "r") as f:
        test_inds = f.readlines()
    item_size = len(test_inds)

    if int(year) == 2007:
        for ind in test_inds:
            ind = ind.strip()
            img = Image.open("{}{}.jpg".format(image_path, ind))
            annotation = read_content("{}{}.xml".format(annot_path, ind))
            dataset.append((img, annotation))

    if int(year) == 2012:
        for ind in test_inds:
            ind = ind.strip()
            img = Image.open("{}{}.jpg".format(image_path, ind))
            dataset.append(img)

    return dataset


def trans_voc(dataset, augment=False, additional_channel=None):
    """
    Transform voc dataset into torch-style data.
    :param dataset
    :param augment:
    :param additional_channel: None or [year (2007, 2012), mode ('canny', laplacian')],
                               such as [2007, 'canny'].
    :return:
    """
    trans_dataset = {}
    data_size = len(dataset)

    load_num = 0
    for ind, (raw_img, raw_annota) in enumerate(dataset):
        # show_sample_img(raw_img, raw_annota)

        # # partial dataset loading.
        # load_num += 1
        # if load_num > 400:
        #     break

        # Stage 1:
        img_ind = raw_annota['annotation']['filename']
        objs = raw_annota['annotation']['object']
        # img_w = annota['annotation']['size']['width']
        # img_h = annota['annotation']['size']['height']
        # img_c = annota['annotation']['size']['depth']

        if type(objs) == list:
            n_objs = len(objs)
            objs_lst = []
            for i in range(n_objs):
                label = labels2val(objs[i]['name'])
                # x_min, y_min, x_max, y_max
                bbox = [int(objs[i]['bndbox']['xmin']),
                        int(objs[i]['bndbox']['ymin']),
                        int(objs[i]['bndbox']['xmax']),
                        int(objs[i]['bndbox']['ymax'])]
                bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
                objs_lst.append(np.asarray([label] + bbox))
        elif type(objs) == dict:
            n_objs = 1
            label = labels2val(objs['name'])
            # x_min, y_min, x_max, y_max
            bbox = [int(objs['bndbox']['xmin']),
                    int(objs['bndbox']['ymin']),
                    int(objs['bndbox']['xmax']),
                    int(objs['bndbox']['ymax'])]
            bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
            objs_lst = [np.asarray([label] + bbox)]
        else:
            raise LookupError

        # Stage 2:
        img_ind = img_ind
        img = raw_img
        objs = objs_lst
        transform_params = []

        # ---------- Images ----------
        # if ind % 200 == 0:
        #     raw_image = img
        #     raw_image.show()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img.convert('RGB'))
        if additional_channel:
            add_c = get_additional_channel("data/VOC/VOCdevkit/VOC{}/JPEGImages/{}".
                                           format(additional_channel[0], img_ind), additional_channel[1])
            add_c = transforms.ToTensor()(add_c)
            img = torch.cat((img, add_c), 0)  # 4 channels

        # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     print("++++++++++ images with less than three channels ++++++++++")
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape  # c, h, w

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        transform_params += pad
        # if ind % 200 == 0:
        #     raw_pad_image, raw_pad = padding_square_np(raw_image, 0)
        # print("img size -- {}".format(img.shape))
        # if ind % 200 == 0:
        #     raw_pad_image.show()
        #     # print("raw_image size -- {}".format(raw_pad_image.size))

        _, padded_h, padded_w = img.shape
        assert padded_h == padded_w
        transform_params.append(padded_h)

        # ---------- Labels ----------
        labels = np.stack(objs, 0)  # [[label, x_center, y_center, width, height], ...]

        boxes = torch.from_numpy(labels)
        # [label, x_center, y_center, width, height] --> [label, x_min, y_min, x_max, y_max]
        x1 = (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = (boxes[:, 2] + boxes[:, 4] / 2)

        # if ind % 200 == 0:
        #     # Visualize bbox in the raw_image.
        #     raw_boxes = np.zeros((len(objs), 4))
        #     raw_boxes[:, 0] = x1.numpy()
        #     raw_boxes[:, 1] = y1.numpy()
        #     raw_boxes[:, 2] = x2.numpy()
        #     raw_boxes[:, 3] = y2.numpy()
        #     print("cls labels of img {}".format(img_ind))
        #     for i in range(len(labels)):
        #         print(val2labels(labels[i, 0]))
        #     for ind in range(raw_boxes.shape[0]):
        #         i_box = list(raw_boxes[ind, :])
        #         bbox_draw = ImageDraw.Draw(raw_image)
        #         bbox_draw.rectangle(i_box, outline='red')
        #     raw_image.show()

        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]

        # if ind % 200 == 0:
        #     # Visualize bbox in the raw_pad_image.
        #     raw_boxes = np.zeros((len(objs), 4))
        #     raw_boxes[:, 0] = x1.numpy()
        #     raw_boxes[:, 1] = y1.numpy()
        #     raw_boxes[:, 2] = x2.numpy()
        #     raw_boxes[:, 3] = y2.numpy()
        #     for ind in range(raw_boxes.shape[0]):
        #         i_box = list(raw_boxes[ind, :])
        #         bbox_draw = ImageDraw.Draw(raw_pad_image)
        #         bbox_draw.rectangle(i_box, outline='red')
        #     raw_pad_image.show()

        # [x_min, y_min, x_max, y_max] --> [x_center, y_center, width, height]
        boxes[:, 1] = (x1 + x2) / 2
        boxes[:, 2] = (y1 + y2) / 2
        boxes[:, 3] = x2 - x1
        boxes[:, 4] = y2 - y1

        # Rescale bbox while resize the image to 412*412
        boxes[:, 1:] = boxes[:, 1:] * (416 / padded_w)

        # Assign bbox to a cell, one cell of 13 * 13 cells, as what yolo does.
        cell_x, cell_y = 416 / 13, 416 / 13
        c_x = torch.floor(boxes[:, 1] / cell_x)
        c_y = torch.floor(boxes[:, 2] / cell_y)
        # # valid check (not necessary normally)
        # valid_check = torch.ones(c_x.shape)
        # valid_check = valid_check.new_full(c_x.shape, 12)
        # c_x = torch.min(valid_check, c_x.float())  # In case for box[:, 1] == 416
        # c_y = torch.min(valid_check, c_y.float())  # In case for box[:, 2] == 416

        # tar_boxes is used for evaluation.
        # [t_conf, t_cls, x, y, w, h] (at present) --> [batch_ind, t_cls, x, y, w, h] (collate_fn())
        tar_boxes = torch.zeros(len(boxes), 6)
        tar_boxes[:, 1:] = boxes
        tar_boxes[:, 0] = 1

        # targets is used for training.
        targets = torch.zeros(1, 13, 13, 26)
        # [t_conf, t_cls, x, y, w, h, one_hot_t_cls(20)]
        for i in range(len(boxes)):
            targets[:, int(c_x[i]), int(c_y[i]), :6] = tar_boxes[i, :]
            targets[:, int(c_x[i]), int(c_y[i]), int(6 + tar_boxes[i, 1])] = 1
        # print(targets[:, 6, 6, :])

        # Apply augmentations
        if augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        transform_params = torch.from_numpy(np.asarray(transform_params).reshape(1, -1)).float()
        trans_dataset[ind] = {'img': img,
                              'img_ind': img_ind,
                              'targets': targets,
                              'transform_params': transform_params,
                              'tar_boxes': tar_boxes}

    if load_num == 0:
        assert len(trans_dataset) == data_size
    return trans_dataset


def trans_voc_nolabel(dataset, additional_channel=None):
    """
    Transform voc dataset into torch-style data.
    :param dataset
    :param additional_channel: None or [year (2007, 2012), mode ('canny', laplacian')],
                               such as [2007, 'canny'].
    :return:
    """
    trans_dataset = {}
    data_size = len(dataset)

    for ind, raw_img in enumerate(dataset):
        img_ind = img_ind
        img = raw_img
        transform_params = []

        # ---------- Images ----------
        # if ind % 200 == 0:
        #     raw_image = img
        #     raw_image.show()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img.convert('RGB'))
        if additional_channel:
            add_c = get_additional_channel("data/VOC/VOCdevkit/VOC{}/JPEGImages/{}".
                                           format(additional_channel[0], img_ind), additional_channel[1])
            add_c = transforms.ToTensor()(add_c)
            img = torch.cat((img, add_c), 0)  # 4 channels

        # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     print("++++++++++ images with less than three channels ++++++++++")
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape  # c, h, w

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        transform_params += pad
        # if ind % 200 == 0:
        #     raw_pad_image, raw_pad = padding_square_np(raw_image, 0)
        # print("img size -- {}".format(img.shape))
        # if ind % 200 == 0:
        #     raw_pad_image.show()
        #     # print("raw_image size -- {}".format(raw_pad_image.size))

        _, padded_h, padded_w = img.shape
        assert padded_h == padded_w
        transform_params.append(padded_h)
        transform_params = torch.from_numpy(np.asarray(transform_params).reshape(1, -1)).float()
        trans_dataset[ind] = {'img': img,
                              'img_ind': img_ind,
                              'transform_params': transform_params}

    assert len(trans_dataset) == data_size
    return trans_dataset


class ListDataset(Dataset):
    def __init__(self, dataset_dict, img_size=416):
        self.dataset = dataset_dict
        self.img_size = img_size
        self.batch_count = 0

    def __getitem__(self, index):
        return self.dataset[index]['img_ind'], \
               self.dataset[index]['img'], \
               self.dataset[index]['targets'], \
               self.dataset[index]['transform_params'], \
               self.dataset[index]['tar_boxes']

    def collate_fn(self, batch):
        """ merges a list of samples to form a mini-batch. """
        img_ind, imgs, targets, transform_params, tar_boxes = list(zip(*batch))
        targets = torch.cat(targets, 0)  # [t_conf, t_cls, x, y, w, h, one_hot_t_cls(20)]
        imgs = torch.stack([square_resize(img, self.img_size) for img in imgs])
        transform_params = torch.cat(transform_params, 0)
        # Remove empty placeholder targets
        tar_boxes = [boxes for boxes in tar_boxes if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(tar_boxes):
            boxes[:, 0] = i
        tar_boxes = torch.cat(tar_boxes, 0)  # [batch_ind, t_cls, x, y, w, h]
        self.batch_count += 1
        # imgs: [N_B, 3, 416, 416]
        # targets: [N_B, 13, 13, 5]
        return img_ind, imgs, targets, transform_params, tar_boxes

    def __len__(self):
        return len(self.dataset)


def dimension_cluster(dataset, dataset_des, save_path):
    """
    Do dimension cluster, i.e., calculate the mean (x, y, w, h) for every cell.
    :param dataset:
    :param dataset_des:
    :param save_path:
    :return:
    """
    init_bbox = np.zeros((4, 13, 13))
    x_channel = np.indices((13, 13))[0]
    y_channel = np.indices((13, 13))[1]
    w_channel = np.full((13, 13), fill_value=3)
    w_channel[:, 0] = 1
    w_channel[0, :] = 1
    w_channel[:, 12] = 1
    w_channel[12, :] = 1
    h_channel = w_channel
    cell_x, cell_y = 416 / 13, 416 / 13
    init_bbox[0, :, :] = cell_x * x_channel + cell_x / 2
    init_bbox[1, :, :] = cell_y * y_channel + cell_y / 2
    init_bbox[2, :, :] = cell_x * w_channel
    init_bbox[3, :, :] = cell_y * h_channel

    pre_bbox = torch.from_numpy(init_bbox).permute(1, 2, 0).to("cuda").float()
    counter = torch.ones((13, 13, 1)).int().to("cuda")  # counter the number of objects of a cell.

    # print(pre_bbox)
    for item in dataset:
        targets = dataset[item]['targets']  # [t_conf, t_cls, x, y, w, h, one_hot_t_cls(20)]
        targets = targets.squeeze(0)[:, :, :6]  # (13, 13, 6) -- [t_conf, t_cls, x, y, w, h]
        object_mask = targets[:, :, 0].byte().to("cuda")
        counter[:, :, 0] += object_mask.int()
        # print(counter)
        obj_targets = targets[object_mask]
        obj_targets = obj_targets[:, 2:].to("cuda")
        # obj_pre_bbox = pre_bbox[object_mask]
        # print(obj_targets)
        # print(obj_pre_bbox)
        # print(pre_bbox)
        # print(pre_bbox[object_mask])
        # Moving average
        # print(item)
        # if item == 1085:
        #     # if counter's type was uint8(.byte) instead of int32(.int).
        #     # then, counter number that larger than 255 would cause overflow, i.e., nan.
        #     print(item)
        # print(pre_bbox[object_mask])
        sub = pre_bbox[object_mask] / counter[object_mask].float()
        plus = obj_targets / counter[object_mask].float()
        pre_bbox[object_mask] = pre_bbox[object_mask] - sub + plus
        # print(pre_bbox)
        # print(pre_bbox[object_mask])

    pre_bbox = pre_bbox.permute(2, 0, 1).cpu().numpy()
    for batch_size in [1, 4, 8, 16]:
        batch_size = int(batch_size)
        bbox_np_batch = np.zeros((batch_size, 4, 13, 13))
        for i in range(batch_size):
            bbox_np_batch[i, :, :, :] = pre_bbox
        path = "{}pre_bbox_b_{}_{}_{}.npy".format(save_path, batch_size, dataset_des[0], dataset_des[1])
        np.save(exist_or_create_folder(path), bbox_np_batch)


# def trans_voc_old(root, year, img_set):
#     """
#     Transform voc dataset into torch-style data.
#     :param root:
#     :param year: 2007, 2012.
#     :param img_set: train, trainval, val.
#     :return:
#     """
#     trans_dataset = {}
#     dataset = torchvision.datasets.VOCDetection(root=root, year=year, image_set=img_set,
#                                                 transforms=None,
#                                                 download=True)
#
#     data_size = len(dataset)
#     for img, annota in dataset:
#         # show_sample_img(img, annota)
#         img_ind = annota['annotation']['filename']
#         objs = annota['annotation']['object']
#         img_w = annota['annotation']['size']['width']
#         img_h = annota['annotation']['size']['height']
#         img_c = annota['annotation']['size']['depth']
#
#         if type(objs) == list:
#             n_objs = len(objs)
#             objs_lst = []
#             for i in range(n_objs):
#                 label = labels2val(objs[i]['name'])
#                 bbox = objs[i]['bndbox'].values()  # x_min, y_min, x_max, y_max
#                 bbox = [item for item in map(int, bbox)]
#                 bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
#                 objs_lst.append(np.asarray([label] + bbox))
#         elif type(objs) == dict:
#             n_objs = 1
#             label = labels2val(objs['name'])
#             bbox = objs['bndbox'].values()  # x_min, y_min, x_max, y_max
#             bbox = [item for item in map(int, bbox)]
#             bbox = xyxy2xywh(bbox)  # x_center, y_center, width, height
#             objs_lst = [np.asarray([label] + bbox)]
#         else:
#             raise LookupError
#
#         # img = np.moveaxis(np.array(img), -1, 0)  # PIL.image -> numpy array in shape [C, H, W]
#         # img = padding_square(img)
#         trans_dataset[img_ind] = {'img': img,
#                                   'n_objs': n_objs,
#                                   'objs': objs_lst}
#
#     assert len(trans_dataset) == data_size
#     return trans_dataset


def main():
    global dev
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torchvision.datasets.VOCDetection(root='../data/VOC/',
                                                year='2007',
                                                image_set='train',
                                                transforms=None,
                                                download=True)
    dataset = trans_voc(dataset)
    dimension_cluster(dataset, ['2007', 'train'], "../config/bbox/")


if __name__ == "__main__":
    main()
