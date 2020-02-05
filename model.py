"""
Build the model.
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

import torchvision.models as models
from utils.utils import build_targets, parse_model_config


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration -- module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    bn_momentum = float(hyperparams["bn_momentum"])
    bn_eps = float(hyperparams["bn_eps"])
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # Resnet backbone
        if module_def["type"] == "resnet":
            num = int(module_def["num"])
            pretrained = bool(int(module_def["pretrained"]))
            if num == 18:
                backbone = models.resnet18(pretrained)
                filters = 512
            elif num == 34:
                backbone = models.resnet34(pretrained)
                filters = 512
            elif num == 50:
                backbone = models.resnet50(pretrained)
                filters = 2048
            elif num == 101:
                backbone = models.resnet101(pretrained)
                filters = 2048
            elif num == 152:
                backbone = models.resnet152(pretrained)
                filters = 2048
            else:
                raise AttributeError("Invalid resnet-{} backbone".format(num))

            # print(backbone)
            # Remove the AvgPool & FC layers
            backbone_strip = nn.Sequential(*(list(backbone.children())[:-2]))
            # print(backbone_strip)
            modules.add_module(f"resnet_{num}", backbone_strip)

        # Conv layer
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=bn_momentum, eps=bn_eps))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        # Maxpooling layer
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        # Skip connection layer
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        # Localization or Classification Conv layer
        elif module_def["type"] == "cls_conv":
            layer_i = int(module_def["from"])
            if layer_i != 0:
                input_channels = output_filters[layer_i]
            else:
                input_channels = output_filters[-1]

            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            atv = module_def["activation"]
            pad = (kernel_size - 1) // 2
            # pad = int(module_def["pad"])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=bn_momentum, eps=bn_eps))
            if atv == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif atv == "relu":
                modules.add_module(f"relu_{module_i}", nn.ReLU())

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """ Placeholder for 'route' and 'shortcut' layers. """
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Darknet(nn.Module):
    """ DQNYOLO object detection model. """
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.no_object_loss_scale = 0.1
        # trade-off params on loss calculation.
        self.obj_scale = 1  # trade off when calculate loss_conf
        self.noobj_scale = 0.5  # trade off when calculate loss_conf
        self.conf_scale = 1  # trade off when calculate loss_conf
        self.cls_scale = 1  # trade off when calculate loss_conf

    def forward(self, x, cls_targets=None):
        layer_outputs, dqnyolo_outputs = [], []
        backbone_ind = -1

        if cls_targets is not None:
            tar_conf, tar_cls, obj_mask, no_obj_mask = build_targets(cls_targets)

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if module_def["type"] == "resnet":
                x = module(x)

            if module_def["type"] in ["convolutional", "maxpool"]:
                x = module(x)

            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif module_def["type"] == "cls_conv":
                layer_i = int(module_def["from"])
                if layer_i != 0:
                    # x = layer_outputs[layer_i]
                    x = layer_outputs[backbone_ind]
                # print('input_x_shape', x.shape)
                x = module(x)
                # print('output_x_shape', x.shape)

                # Calculate cls_loss
                out = int(module_def["out"])
                if out == 1:
                    conf_cls_output_ind = len(layer_outputs)
                if out and cls_targets is not None:
                    # print("x.size()")
                    # print(x.size())
                    pred_conf_cls = x.permute(0, 2, 3, 1)
                    pred_conf = pred_conf_cls[:, :, :, 0]
                    pred_conf = torch.sigmoid(pred_conf)
                    pred_cls = pred_conf_cls[:, :, :, 1:]
                    pred_cls = torch.sigmoid(pred_cls)
                    # print("pred_conf")
                    # print(pred_conf)
                    # print(pred_conf.size())
                    # print("tar_conf")
                    # print(tar_conf)
                    # print(tar_conf.size())
                    # print("obj_mask")
                    # print(obj_mask)
                    # print(obj_mask.size())
                    # print("pred_conf[obj_mask]")
                    # print(pred_conf[obj_mask])
                    # print(pred_conf[obj_mask].size())
                    # print("tar_conf[obj_mask]")
                    # print(tar_conf[obj_mask])
                    # print(tar_conf[obj_mask].size())
                    loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tar_conf[obj_mask])
                    # print("no_obj_mask")
                    # print(no_obj_mask)
                    # print(no_obj_mask.size())
                    # print("pred_conf[no_obj_mask]")
                    # print(pred_conf[no_obj_mask])
                    # print(pred_conf[no_obj_mask].size())
                    # print("tar_conf[no_obj_mask]")
                    # print(tar_conf[no_obj_mask])
                    # print(tar_conf[no_obj_mask].size())
                    loss_conf_noobj = self.bce_loss(pred_conf[no_obj_mask], tar_conf[no_obj_mask])
                    # print(pred_conf[0, :, :])
                    # print(tar_conf[0, :, :])
                    # print("\nloss_conf_obj: ", self.obj_scale * loss_conf_obj.item())
                    # print("loss_conf_noobj: ", self.noobj_scale * loss_conf_noobj.item())
                    loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                    # print("tar_cls")
                    # print(tar_cls)
                    # print(tar_cls.size())
                    # print("pred_cls")
                    # print(pred_cls)
                    # print(pred_cls.size())
                    # print("obj_mask")
                    # print(obj_mask)
                    # print(obj_mask.size())
                    # print("pred_cls[obj_mask]")
                    # print(pred_cls[obj_mask])
                    # print(pred_cls[obj_mask].size())
                    # print("tar_cls[obj_mask]")
                    # print(tar_cls[obj_mask])
                    # print(tar_cls[obj_mask].size())
                    # print(pred_cls[0, 6, 6, :])
                    # print(tar_cls[0, 6, 6, :])
                    loss_cls = self.bce_loss(pred_cls[obj_mask], tar_cls[obj_mask])
                    # print("\nloss_conf: ", self.conf_scale * loss_conf.item())
                    # print("loss_cls: ", self.cls_scale * loss_cls.item())
                    loss_conf_cls = self.conf_scale * loss_conf + self.cls_scale * loss_cls
                    # print("\nloss_loc: ", self.loc_scale * loss_loc.item())
                    # print("loss_conf_cls: ", self.conf_cls_scale * loss_conf_cls.item())

            layer_outputs.append(x)
            # print('layer_outputs', i, len(layer_outputs))

        dqnyolo_conf_cls_outputs = layer_outputs[conf_cls_output_ind]

        if cls_targets is None:
            return dqnyolo_conf_cls_outputs
        else:
            return loss_conf_cls, dqnyolo_conf_cls_outputs

    def load_darknet_weights(self, weights_path):
        """
        Parses and loads the weights stored in 'weights_path'
        copied from https://github.com/eriklindernoren/PyTorch-YOLOv3
        """
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 74

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        copied from https://github.com/eriklindernoren/PyTorch-YOLOv3
        :param path: path of the new weights file
        :param cutoff: save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        :return:
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
