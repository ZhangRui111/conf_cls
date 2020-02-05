"""
Object detection environment.
"""
import torch
import torchvision
import numpy as np
import pathlib

from utils.utils import bbox_iou, bbox_iou_multi_dims
from utils.datasets import dimension_cluster, trans_voc


class MemoryDict(object):
    def __init__(self, size, batch_size):
        self.memory_size = size
        self.batch_size = batch_size
        self.memory_dict = {}
        self.memory_counter = 0

    def add_transition(self, input_img, current_bbox, cls_targets, loc_targets, acts):
        """
        Save transitions.
        :param input_img:
        :param current_bbox:
        :param cls_targets:
        :param loc_targets:
        :param acts:
        :return:
        """
        for i in range(self.batch_size):
            new_transition = {
                "input_img": input_img[i, ...].unsqueeze(0),
                "current_bbox": current_bbox[i, ...].unsqueeze(0),
                "cls_targets": cls_targets[i, ...].unsqueeze(0),
                "loc_targets": loc_targets[i, ...].unsqueeze(0),
                "acts": acts[i, ...].unsqueeze(0)
            }
            index = self.memory_counter % self.memory_size
            self.memory_dict[index] = new_transition
            self.memory_counter += 1
        # print("The size of memory_dict ", len(self.memory_dict))

    def sample_batch(self, batch_size=None):
        """
        Sample transitions uniformly.
        :param batch_size:
        :return:
        """
        if batch_size is None:
            batch_size = self.batch_size
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=batch_size)
        input_img_lst, current_bbox_lst, cls_targets_lst, loc_targets_lst, acts_lst = [], [], [], [], []
        for i in sample_index:
            input_img_lst.append(self.memory_dict[i]["input_img"])
            current_bbox_lst.append(self.memory_dict[i]["current_bbox"])
            cls_targets_lst.append(self.memory_dict[i]["cls_targets"])
            loc_targets_lst.append(self.memory_dict[i]["loc_targets"])
            acts_lst.append(self.memory_dict[i]["acts"])
        input_img = torch.cat(input_img_lst, dim=0)
        current_bbox = torch.cat(current_bbox_lst, dim=0)
        cls_targets = torch.cat(cls_targets_lst, dim=0)
        loc_targets = torch.cat(loc_targets_lst, dim=0)
        acts = torch.cat(acts_lst, dim=0)
        return input_img, current_bbox, cls_targets, [loc_targets, acts]


class ObjDetEnv(object):
    """
    Object detection environment.
    """
    def __init__(self, img, label, batch_size, dataset_des=None, dim_cluster=True, iou_threshold=0.5, memory=None):
        # imgs: [N_B, 3, 416, 416]
        # label: [N_B, 13, 13, 6] -- 6: [t_conf, t_cls, x, y, w, h]
        if dataset_des is None:
            self.dataset_des = ['2007', 'train']
        else:
            self.dataset_des = dataset_des
        self.dim_cluster = dim_cluster
        self.iou_threshold = iou_threshold  # iou threshold for trigger action
        self.trans_scale = 0.1  # transform scale
        self.batch_size = batch_size
        self.memory_size = 2000
        if memory is None:
            self.memory = MemoryDict(self.memory_size, batch_size)
        else:
            self.memory = memory

        self.img = img
        self.label = label
        # print(label[0, 6, 6, :])  # tensor([  1.0000,  11.0000, 210.9120, 222.1440, 162.2400, 144.7680])
        # self.label_dict = self.get_label_dict(label)
        self.init_bbox = self.get_init_bbox(batch_size, "config/bbox/", self.dim_cluster)
        self.bbox = self.get_init_bbox(batch_size, "config/bbox/", self.dim_cluster)
        self.object_mask, self.no_object_mask = self.get_object_mask()
        self.done = torch.zeros((self.batch_size, 13, 13)).byte().to('cuda')
        self.steps = 0
        self.max_step = 25
        self.detected_r = 1
        self.undetected_r = -1
        self.iou_increase_r = 0.1
        self.iou_decrease_r = -0.1
        self.exceed_step_p = -2
        self.invalid_p = -0.1
        self.cell_x_min, self.cell_x_max, self.cell_y_min, self.cell_y_max = self.get_cell_limits(batch_size)

    def step(self, actions):
        """
        Env step.
        :param actions: 0 --> translation trigger
                        1 --> translation left
                        2 --> translation right
                        3 --> translation up
                        4 --> translation down
                        5 --> translation fatter
                        6 --> translation thinner
                        7 --> translation taller
                        8 --> translation shorter
        :return:
        """
        # # Initialize rewards
        rewards = torch.zeros((self.batch_size, 13, 13)).float().to('cuda')
        # # All action-masks
        trigger_acts = (actions == 0).byte().to('cuda')
        left_acts = (actions == 1).byte().to('cuda')
        right_acts = (actions == 2).byte().to('cuda')
        up_acts = (actions == 3).byte().to('cuda')
        down_acts = (actions == 4).byte().to('cuda')
        fatter_acts = (actions == 5).byte().to('cuda')
        thinner_acts = (actions == 6).byte().to('cuda')
        taller_acts = (actions == 7).byte().to('cuda')
        shorter_acts = (actions == 8).byte().to('cuda')
        acts_holder = [trigger_acts, left_acts, right_acts, up_acts, down_acts,
                       fatter_acts, thinner_acts, taller_acts, shorter_acts]
        # # Update the self.done signal.
        self.done[trigger_acts] = 1

        # # Update all bboxes
        tmp_bbox = self.bbox.detach().clone().permute(0, 2, 3, 1)

        wh_step_minimum = torch.zeros(self.batch_size, 13, 13).to("cuda") + 5

        for i, acts_mask in enumerate(acts_holder):
            if len(self.label[:, :, :, 2:][acts_mask]) > 0:
                tar_bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                tar_bbox[acts_mask] = self.label[:, :, :, 2:][acts_mask]
                bbox[acts_mask] = tmp_bbox[acts_mask]
                # a = self.label[:, :, :, 2:][acts_mask]
                # b = tmp_bbox[acts_mask]

                # # Debug segmentation.
                # if acts_mask[0, 6, 6] == 1 and i == 0:
                #     print("Break point here!")

                # --------------------------------------------------------- #
                # case 1: constant value
                # case 2: proportional to the width and height
                # case 3: proportional to the width and height but decay by steps.
                # case 4: "x, y" use case 1, "w, h" use case 2 or 3.
                # --------------------------------------------------------- #
                # trans_x_variation or trans_y_variation, at least 3 pixels.
                trans_x_step = max(8 * (0.9 ** self.steps), 3)
                trans_y_step = max(8 * (0.9 ** self.steps), 3)
                # x_variation or y_variation, at least 5 pixels.
                w_step = torch.max(bbox[:, :, :, 2] * self.trans_scale, wh_step_minimum)
                h_step = torch.max(bbox[:, :, :, 3] * self.trans_scale, wh_step_minimum)

                if i == 0:
                    # # Reward for trigger action.
                    rewards[acts_mask] = self.undetected_r
                    iou = bbox_iou_multi_dims(bbox, tar_bbox)
                    # print(iou)
                    detected_mask = (iou > self.iou_threshold).byte()
                    # print(acts_mask)
                    # print(detected_mask)
                    # print(rewards[0, :, :])
                    # print(acts_mask & detected_mask)
                    rewards[acts_mask & detected_mask] = self.detected_r
                    # print(rewards[0, :, :])
                else:
                    # Reward for other actions.
                    iou = bbox_iou_multi_dims(bbox, tar_bbox)

                    if i == 1:
                        # print(bbox[:, :, :, 0])
                        bbox[:, :, :, 0] -= trans_x_step
                        # print(bbox[:, :, :, 0][acts_mask])
                        # print(self.cell_x_min[:, :, :, 0][acts_mask])
                        valid_cell_mask = (bbox[:, :, :, 0] >= self.cell_x_min[:, :, :, 0]).byte()
                        # print(valid_cell_mask)

                        # if self.steps % 10 == 0:
                        #     print("steps: {} -- trans_x_variation: {}"
                        #           .format(self.steps, trans_x_variation))
                    elif i == 2:
                        bbox[:, :, :, 0] += trans_x_step
                        # print(self.cell_x_max[acts_mask])
                        valid_cell_mask = (bbox[:, :, :, 0] <= self.cell_x_max[:, :, :, 0]).byte()
                    elif i == 3:
                        # print(bbox[:, :, :, 1])
                        bbox[:, :, :, 1] -= trans_y_step
                        # print(bbox[:, :, :, 1])
                        # print(self.cell_y_min[acts_mask])
                        valid_cell_mask = (bbox[:, :, :, 1] >= self.cell_y_min[:, :, :, 0]).byte()
                        # print(valid_cell_mask)
                    elif i == 4:
                        bbox[:, :, :, 1] += trans_y_step
                        # print(self.cell_y_max[acts_mask])
                        valid_cell_mask = (bbox[:, :, :, 1] <= self.cell_y_max[:, :, :, 0]).byte()
                    elif i == 5:
                        # print(bbox[:, :, :, 2])
                        bbox[:, :, :, 2] += w_step
                        # print(bbox[:, :, :, 2])
                        valid_cell_mask = (bbox[:, :, :, 2] <= 400).byte()
                        # print(valid_cell_mask)

                        # if self.steps % 10 == 0:
                        #     print("steps: {} -- x_variation {}"
                        #           .format(self.steps, x_variation[acts_mask]))
                    elif i == 6:
                        bbox[:, :, :, 2] -= w_step
                        valid_cell_mask = (bbox[:, :, :, 2] >= 1).byte()
                    elif i == 7:
                        bbox[:, :, :, 3] += h_step
                        valid_cell_mask = (bbox[:, :, :, 3] <= 400).byte()
                    elif i == 8:
                        bbox[:, :, :, 3] -= h_step
                        valid_cell_mask = (bbox[:, :, :, 3] >= 1).byte()
                    else:
                        raise ValueError("Invalid action code!")

                    invalid_cell_mask = ~valid_cell_mask
                    update_mask = acts_mask & valid_cell_mask
                    # print(update_mask)
                    # print(acts_mask)
                    # print(valid_cell_mask)
                    # print(tmp_bbox[update_mask])
                    tmp_bbox[update_mask] = bbox[update_mask]
                    # print(tmp_bbox[update_mask])
                    # print(self.bbox[0, :, :, :])
                    self.bbox = tmp_bbox.permute(0, 3, 1, 2)
                    # print(self.bbox[0, :, :, :])

                    # print(iou)
                    new_bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                    new_bbox[acts_mask] = bbox[acts_mask]
                    new_iou = bbox_iou_multi_dims(new_bbox, tar_bbox)
                    # print(new_iou)
                    iou_incre_mask = (new_iou > iou).byte()
                    # print(iou_incre_mask)
                    # print(acts_mask)
                    rewards[acts_mask] = self.iou_decrease_r
                    rewards[acts_mask & iou_incre_mask] = self.iou_increase_r

                    # Penalty
                    # exceeding max_step penalty
                    if self.steps >= self.max_step:
                        rewards[acts_mask] = self.exceed_step_p
                    # invalid actions penalty
                    rewards[acts_mask & invalid_cell_mask] = self.invalid_p
                    # print(rewards)
            else:
                continue

        # # Rewards for cells without objects:
        # For all cells without objects, all actions have reward == 0.
        rewards[self.no_object_mask] = 0
        # print(rewards)
        # print(rewards[:, 6, 6])

        self.steps += 1
        return rewards, self.done

    def eval_step(self, actions):
        """
        Env step.
        :param actions: 0 --> translation trigger
                        1 --> translation left
                        2 --> translation right
                        3 --> translation up
                        4 --> translation down
                        5 --> translation fatter
                        6 --> translation thinner
                        7 --> translation taller
                        8 --> translation shorter
        :return:
        """
        # # All action-masks
        trigger_acts = (actions == 0).byte().to('cuda')
        left_acts = (actions == 1).byte().to('cuda')
        right_acts = (actions == 2).byte().to('cuda')
        up_acts = (actions == 3).byte().to('cuda')
        down_acts = (actions == 4).byte().to('cuda')
        fatter_acts = (actions == 5).byte().to('cuda')
        thinner_acts = (actions == 6).byte().to('cuda')
        taller_acts = (actions == 7).byte().to('cuda')
        shorter_acts = (actions == 8).byte().to('cuda')
        acts_holder = [trigger_acts, left_acts, right_acts, up_acts, down_acts,
                       fatter_acts, thinner_acts, taller_acts, shorter_acts]
        # # Update the self.done signal.
        self.done[trigger_acts] = 1

        # # Update all bboxes
        tmp_bbox = self.bbox.detach().clone().permute(0, 2, 3, 1)

        x_variation_minimum = torch.zeros(self.batch_size, 13, 13).to("cuda") + 5

        for i, acts_mask in enumerate(acts_holder):
            if len(self.label[:, :, :, 2:][acts_mask]) > 0:
                tar_bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                tar_bbox[acts_mask] = self.label[:, :, :, 2:][acts_mask]
                bbox[acts_mask] = tmp_bbox[acts_mask]
                # a = self.label[:, :, :, 2:][acts_mask]
                # b = tmp_bbox[acts_mask]

                # --------------------------------------------------------- #
                # case 1: constant value
                # case 2: proportional to the width and height
                # case 3: proportional to the width and height but decay by steps.
                # case 4: "x, y" use case 1, "w, h" use case 2 or 3.
                # --------------------------------------------------------- #
                # trans_x_variation or trans_y_variation, at least 3 pixels.
                trans_x_variation = max(8 * (0.9 ** self.steps), 3)
                trans_y_variation = max(8 * (0.9 ** self.steps), 3)
                # x_variation or y_variation, at least 5 pixels.
                x_variation = torch.max(bbox[:, :, :, 2] * self.trans_scale, x_variation_minimum)
                y_variation = torch.max(bbox[:, :, :, 2] * self.trans_scale, x_variation_minimum)

                if i == 0:
                    pass
                else:
                    # iou = bbox_iou_multi_dims(bbox, tar_bbox)

                    if i == 1:
                        bbox[:, :, :, 0] -= trans_x_variation
                        valid_cell_mask = (bbox[:, :, :, 0] >= self.cell_x_min[:, :, :, 0]).byte()
                    elif i == 2:
                        bbox[:, :, :, 0] += trans_x_variation
                        valid_cell_mask = (bbox[:, :, :, 0] <= self.cell_x_max[:, :, :, 0]).byte()
                    elif i == 3:
                        bbox[:, :, :, 1] -= trans_y_variation
                        valid_cell_mask = (bbox[:, :, :, 1] >= self.cell_y_min[:, :, :, 0]).byte()
                    elif i == 4:
                        bbox[:, :, :, 1] += trans_y_variation
                        valid_cell_mask = (bbox[:, :, :, 1] <= self.cell_y_max[:, :, :, 0]).byte()
                    elif i == 5:
                        bbox[:, :, :, 2] += x_variation
                        valid_cell_mask = (bbox[:, :, :, 2] <= 400).byte()
                    elif i == 6:
                        bbox[:, :, :, 2] -= x_variation
                        valid_cell_mask = (bbox[:, :, :, 2] >= 1).byte()
                    elif i == 7:
                        bbox[:, :, :, 3] += y_variation
                        valid_cell_mask = (bbox[:, :, :, 3] <= 400).byte()
                    elif i == 8:
                        bbox[:, :, :, 3] -= y_variation
                        valid_cell_mask = (bbox[:, :, :, 3] >= 1).byte()
                    else:
                        raise ValueError("Invalid action code!")

                    update_mask = acts_mask & valid_cell_mask
                    tmp_bbox[update_mask] = bbox[update_mask]
                    self.bbox = tmp_bbox.permute(0, 3, 1, 2)

                    # # print(iou)
                    # new_bbox = torch.zeros(self.batch_size, 13, 13, 4).to("cuda")
                    # new_bbox[acts_mask] = bbox[acts_mask]
                    # new_iou = bbox_iou_multi_dims(new_bbox, tar_bbox)
                    # # print(new_iou)
                    # iou_incre_mask = (new_iou > iou).byte()
                    # print(iou_incre_mask)
            else:
                continue

        self.steps += 1
        return self.done

    def reset(self, img, label, iou_threshold=0.5):
        self.__init__(img, label, self.batch_size, self.dataset_des, self.dim_cluster, iou_threshold, self.memory)

    def get_objects_rewards(self):
        """
        Final mean reward per object cell and
        the ratio of detected objects to all objects.
        """
        num_objects = torch.sum(self.object_mask).item()
        # print(num_objects)
        f_object_mask = self.object_mask.detach().clone().byte()
        f_bbox = self.bbox.detach().clone().permute(0, 2, 3, 1)
        f_label = self.label.detach().clone()[:, :, :, 2:]
        f_label_obj_mask = f_label[f_object_mask]
        f_bbox_obj_mask = f_bbox[f_object_mask]
        objects_r = bbox_iou_multi_dims(f_label_obj_mask, f_bbox_obj_mask)
        # print(objects_r)
        # print(objects_r > self.iou_threshold)
        return torch.sum(objects_r).float() / num_objects, \
               torch.sum((objects_r > self.iou_threshold)).float() / num_objects

    def get_img(self):
        return self.img

    def get_label(self):
        return self.label

    def get_bbox(self, norm_bbox=False):
        if norm_bbox:
            norm_bbox = self.bbox.detach().clone().permute(0, 2, 3, 1)
            # # Normalize the x and y
            # print(norm_bbox[:, :4, :4, 0])
            # print(norm_bbox[:, :4, :4, 1])
            # print(self.cell_x_min.squeeze(3)[:, :4, :4])
            # print(self.cell_y_min.squeeze(3)[:, :4, :4])
            norm_bbox[:, :, :, 0] = norm_bbox[:, :, :, 0] - self.cell_x_min.squeeze(3)
            norm_bbox[:, :, :, 1] = norm_bbox[:, :, :, 1] - self.cell_y_min.squeeze(3)
            # print(norm_bbox[:, :4, :4, 0])
            # print(norm_bbox[:, :4, :4, 1])
            norm_bbox[:, :, :, 0] = norm_bbox[:, :, :, 0] / 32
            norm_bbox[:, :, :, 1] = norm_bbox[:, :, :, 1] / 32
            # print(norm_bbox[:, :4, :4, 0])
            # print(norm_bbox[:, :4, :4, 1])
            # # Normalize the width and height
            # print(norm_bbox[:, :4, :4, 2])
            # print(norm_bbox[:, :4, :4, 3])
            norm_bbox[:, :, :, 2:] = norm_bbox[:, :, :, 2:] / 416
            # print(norm_bbox[:, :4, :4, 2])
            # print(norm_bbox[:, :4, :4, 3])
            return norm_bbox.permute(0, 3, 1, 2)
        else:
            return self.bbox

    def get_init_bbox(self, batch_size, path, cluster=True):
        """ Set initial visible bbox for every cell. """
        if not cluster:
            file_path = "./config/bbox/init_bbox_b_{}.npy".format(batch_size)
            if not pathlib.Path(file_path).exists():
                # Generate init_bbox and save it.
                bbox_np = np.zeros((4, 13, 13))
                x_channel = np.indices((13, 13))[0]
                y_channel = np.indices((13, 13))[1]
                w_channel = np.full((13, 13), fill_value=3)
                w_channel[:, 0] = 1
                w_channel[0, :] = 1
                w_channel[:, 12] = 1
                w_channel[12, :] = 1
                h_channel = w_channel
                cell_x, cell_y = 416 / 13, 416 / 13
                bbox_np[0, :, :] = cell_x * x_channel + cell_x / 2
                bbox_np[1, :, :] = cell_y * y_channel + cell_y / 2
                bbox_np[2, :, :] = cell_x * w_channel
                bbox_np[3, :, :] = cell_y * h_channel

                bbox_np_batch = np.zeros((batch_size, 4, 13, 13))
                for i in range(batch_size):
                    bbox_np_batch[i, :, :, :] = bbox_np
                np.save(file_path)

            bbox_np_batch = np.load(file_path)
        else:
            # file_path = "{}pre_bbox_b_{}_{}_{}.npy".format(path, batch_size, self.dataset_des[0], self.dataset_des[1])
            file_path = "{}pre_bbox_b_{}_{}_{}.npy".format(path, batch_size, self.dataset_des[0], 'trainval')
            if not pathlib.Path(file_path).exists():
                dataset = torchvision.datasets.VOCDetection(root='../data/VOC/',
                                                            year=self.dataset_des[0],
                                                            image_set=self.dataset_des[1],
                                                            transforms=None,
                                                            download=True)
                dataset = trans_voc(dataset)
                dimension_cluster(dataset, [self.dataset_des[0], self.dataset_des[1]], path)
            bbox_np_batch = np.load(file_path)

        return torch.from_numpy(bbox_np_batch).float().to('cuda')

    def get_object_mask(self):
        object_mask = self.label[:, :, :, 0].byte().to('cuda')
        no_object_mask = ~object_mask
        return object_mask, no_object_mask

    def get_cell_limits(self, batch_size=8):
        x_channel, y_channel = 32 * np.indices((13, 13))
        cell_x_min = np.stack([x_channel] * batch_size, axis=0)
        cell_x_max = cell_x_min + 32
        cell_y_min = np.stack([y_channel] * batch_size, axis=0)
        cell_y_max = cell_y_min + 32
        return (torch.from_numpy(cell_x_min).float().to("cuda").unsqueeze(3),
                torch.from_numpy(cell_x_max).float().to("cuda").unsqueeze(3),
                torch.from_numpy(cell_y_min).float().to("cuda").unsqueeze(3),
                torch.from_numpy(cell_y_max).float().to("cuda").unsqueeze(3))

    # def get_label_dict(self, label):
    #     # print(label)
    #     label_np = label.clone()
    #     label_np = label_np.cpu().numpy()[:, 0]
    #     unique, counts = np.unique(label_np, return_counts=True)
    #     label_dict_count = dict(zip(unique, counts))
    #     print(label_dict_count)
    #     label_dict = {}
    #     total_count = 0
    #     for ind_img in range(self.batch_size):
    #         for ind in range(label_dict_count[ind_img]):
    #             dict_key = "{}-{}-{}".format(int(ind_img),
    #                                          int(label[total_count, :][1]),
    #                                          int(label[total_count, :][2]))
    #             label_dict[dict_key] = label[total_count, -4:]
    #             total_count += 1
    #     print("dict_key: {}".format(label_dict))
    #     return label_dict

    # def step_for_loop(self, actions):
    #     """
    #     Env step.
    #     :param actions: 0 --> translation trigger
    #                     1 --> translation left
    #                     2 --> translation right
    #                     3 --> translation up
    #                     4 --> translation down
    #                     5 --> translation fatter
    #                     6 --> translation thinner
    #                     7 --> translation taller
    #                     8 --> translation shorter
    #     :return:
    #     """
    #     # print("actions")
    #     # print(actions)
    #     # print("self.no_object_mask")
    #     # print(self.no_object_mask)
    #     rewards = torch.zeros((self.batch_size, 13, 13)).float().to('cuda')
    #     trigger_acts = (actions == 0).byte().to('cuda')
    #     # print("trigger_acts")
    #     # print(trigger_acts)
    #     # self.done = torch.min(torch.add(self.done, trigger_acts), self.full_done).to('cuda')
    #     self.done[trigger_acts] = 1
    #     # print("self.done")
    #     # print(self.done)
    #
    #     # # Rewards for cells without objects:
    #     # # For all cells without objects, any action (except trigger) has reward == -1,
    #     # # except trigger action that has reward == 0.
    #     rewards[self.no_object_mask] = -1
    #     # print("rewards after rewards[self.no_object_mask] = -1")
    #     # print(rewards)
    #     # print("--------------------")
    #     # print(self.no_object_mask)
    #     # print(trigger_acts)
    #     # print((self.no_object_mask & trigger_acts))
    #     rewards[(self.no_object_mask & trigger_acts)] = 0
    #     # print("rewards after rewards[self.done] = 0")
    #     # print(rewards)
    #     # # Rewards for cells without objects are subdued by a factor
    #     # rewards *= self.no_object_r_scale
    #
    #     # # Cells with objects
    #     object_inds = self.object_mask.nonzero()  # [[ind_img, cell_x, cell_y], ...]
    #     for ind in range(len(object_inds)):
    #         item = object_inds[ind]
    #         rewards[item[0]][item[1]][item[2]] = self.step_reward(
    #             self.bbox[item[0], :, item[1], item[2]],
    #             self.label[item[0], item[1], item[2], 2:],
    #             actions[item[0], item[1], item[2]],
    #             item[0],
    #             item[1],
    #             item[2]
    #         )
    #     self.steps += 1
    #     return rewards, self.done
    #
    # def step_reward(self, bbox, tar_bbox, action, b_ind, w_ind, h_ind):
    #     # TODO case: one cell with multiple objects
    #     # TODO case: this is a problem: localization, one cell can only target one object now.
    #     # TODO case: possible solution --> add one dim, so that more than one bboxes each cell.
    #     # TODO case: This problem doesn't affect current pipeline.
    #     bbox = bbox.unsqueeze(0)
    #     tar_bbox = tar_bbox.reshape(1, -1).to('cuda')
    #     iou = bbox_iou(bbox, tar_bbox)
    #     # print("Old IoU {}".format(iou))
    #     if action == 0:
    #         # Reward for trigger action
    #         if iou >= self.iou_threshold:
    #             return 10
    #         else:
    #             return -5
    #     else:
    #         # Reward for other actions
    #         cell_x_min, cell_x_max = (32 * w_ind).float(), (32 * (w_ind + 1)).float()
    #         cell_y_min, cell_y_max = (32 * h_ind).float(), (32 * (h_ind + 1)).float()
    #
    #         # --------------------------------------------------------- #
    #         # case 1: constant value
    #         # case 2: proportional to the width and height
    #         # case 3: proportional to the width and height but decay by steps.
    #         # case 4: "x, y" use case 1, "w, h" use case 2 or 3.
    #         # --------------------------------------------------------- #
    #         # trans_x_variation = 4 * (0.9 ** self.steps)
    #         # trans_y_variation = 4 * (0.9 ** self.steps)
    #         # x_variation = bbox[:, 2] * self.trans_scale * (0.9 ** self.steps)
    #         # y_variation = bbox[:, 2] * self.trans_scale * (0.9 ** self.steps)
    #         # trans_x_variation = max(4 * (0.9 ** self.steps), 2)
    #         # trans_y_variation = max(4 * (0.9 ** self.steps), 2)
    #         # x_variation = max(bbox[:, 2] * self.trans_scale * (0.9 ** self.steps), 2)
    #         # y_variation = max(bbox[:, 2] * self.trans_scale * (0.9 ** self.steps), 2)
    #         trans_x_variation = max(8 * (0.9 ** self.steps), 2)  # At least 2 pixels.
    #         trans_y_variation = max(8 * (0.9 ** self.steps), 2)
    #         x_variation = max(bbox[:, 2] * self.trans_scale, 2)
    #         y_variation = max(bbox[:, 2] * self.trans_scale, 2)
    #         # print("steps: {} -- trans_x_variation: {} -- x_variation {}".format(self.steps, trans_x_variation, x_variation))
    #
    #         if action == 1:
    #             bbox[:, 0] -= trans_x_variation
    #             if bbox[:, 0].squeeze() < cell_x_min:
    #                 return self.invalid_p
    #         elif action == 2:
    #             bbox[:, 0] += trans_x_variation
    #             if bbox[:, 0].squeeze() > cell_x_max:
    #                 return self.invalid_p
    #         elif action == 3:
    #             bbox[:, 1] -= trans_y_variation
    #             if bbox[:, 1].squeeze() < cell_y_min:
    #                 return self.invalid_p
    #         elif action == 4:
    #             bbox[:, 1] += trans_y_variation
    #             if bbox[:, 1].squeeze() > cell_y_max:
    #                 return self.invalid_p
    #         elif action == 5:
    #             bbox[:, 2] += x_variation
    #         elif action == 6:
    #             new_w = bbox[:, 2] - x_variation
    #             bbox[:, 2] = max(0, new_w)
    #         elif action == 7:
    #             bbox[:, 3] += y_variation
    #         elif action == 8:
    #             new_h = bbox[:, 3] - y_variation
    #             bbox[:, 3] = max(0, new_h)
    #         else:
    #             raise ValueError("Invalid action code!")
    #
    #         self.bbox[b_ind, :, w_ind, h_ind] = bbox
    #         new_iou = bbox_iou(bbox, tar_bbox)
    #         # print("New IoU {}".format(new_iou))
    #         if new_iou > iou:
    #             return self.iou_increase_r
    #         else:
    #             return self.trans_bbox_r


def main():
    pass


if __name__ == '__main__':
    main()
