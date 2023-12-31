# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import masked_log_softmax, masked_softmax


# ==================================================================================================================== #
#                                                  Vision Pretraining                                                  #
# ==================================================================================================================== #
def kl_1601(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    if (weight > 0) and (image_cls is not None):
        image_target = image_cls
        loss = nn.KLDivLoss(reduction="none")(F.log_softmax(prediction_scores_v, dim=2), image_target)
        return weight * torch.sum(loss * (label == 1).unsqueeze(2).float()) / max(torch.sum((label == 1)), 1)
    else:
        return 0


def mse_2048(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    # regress the feature
    if (weight > 0) and (image_feat is not None):
        image_target = copy.deepcopy(image_feat)
        loss = nn.MSELoss(reduction="none")(prediction_scores_v, image_target)
        return weight * torch.sum(loss * (label == 1).unsqueeze(2).float()) / \
            max(torch.sum((label == 1).unsqueeze(2).expand_as(loss)), 1)
    else:
        return 0


def nce_2048(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    # NCE loss
    num_negative = 128
    if (weight > 0) and (image_feat is not None):

        image_target = copy.deepcopy(image_feat)

        # generate negative sampled index.
        num_across_batch = int(num_negative * 0.7)
        num_inside_batch = int(num_negative * 0.3)

        batch_size, num_regions, _ = prediction_scores_v.size()
        assert batch_size != 0
        # random negative across batches.
        row_across_index = image_target.new(batch_size, num_regions, num_across_batch).random_(0, batch_size - 1)
        col_across_index = image_target.new(batch_size, num_regions, num_across_batch).random_(0, num_regions)

        for i in range(batch_size - 1):
            row_across_index[i][row_across_index[i] == i] = batch_size - 1
        final_across_index = row_across_index * num_regions + col_across_index

        # random negative inside batches.
        row_inside_index = image_target.new(batch_size, num_regions, num_inside_batch).zero_()
        col_inside_index = image_target.new(batch_size, num_regions, num_inside_batch).random_(0, num_regions - 1)

        for i in range(batch_size):
            row_inside_index[i] = i
        for i in range(num_regions - 1):
            col_inside_index[:, i, :][col_inside_index[:, i, :] == i] = num_regions - 1
        final_inside_index = row_inside_index * num_regions + col_inside_index

        final_index = torch.cat((final_across_index, final_inside_index), dim=2)

        # Let's first sample where we need to compute.
        predict_v = prediction_scores_v[label == 1]
        neg_index_v = final_index[label == 1]

        flat_image_target = image_target.view(batch_size * num_regions, -1)
        # we also need to append the target feature at the beginning.
        negative_v = flat_image_target[neg_index_v]
        positive_v = image_target[label == 1]
        sample_v = torch.cat((positive_v.unsqueeze(1), negative_v), dim=1)

        # calculate the loss.
        score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
        return weight * nn.CrossEntropyLoss()(score, image_target.new(score.size(0)).zero_())
    else:
        return 0


def xent_1600(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    if (weight > 0) and (obj_labels is not None) and (obj_confs is not None):
        # hard object labels
        image_target, mask_conf = obj_labels, obj_confs
        loss = nn.CrossEntropyLoss(reduction='none')(prediction_scores_v.reshape(-1, 1600), image_target.view(-1,))
        loss = loss * mask_conf.view(-1)
        return weight * torch.sum(loss * (label.view(-1) == 1)) / max(torch.sum((label == 1)), 1)
    else:
        return 0


def xent_400(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    if (weight > 0) and (attr_labels is not None) and (attr_confs is not None):
        # hard attribute labels
        image_target, mask_conf = attr_labels, attr_confs
        loss = nn.CrossEntropyLoss(reduction='none')(prediction_scores_v.reshape(-1, 400), image_target.view(-1,))
        loss = loss * mask_conf.view(-1)
        return weight * torch.sum(loss * (label.view(-1) == 1)) / max(torch.sum((label == 1)), 1)
    else:
        return 0


def huber_2048(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    if (weight > 0) and (image_feat is not None):
        # regress the feature
        image_target = copy.deepcopy(image_feat)
        loss = nn.SmoothL1Loss(reduction='none')(prediction_scores_v, image_target)
        return weight * torch.sum(loss * (label == 1).unsqueeze(2).float()) / \
            max(torch.sum((label == 1).unsqueeze(2).expand_as(loss)), 1)
    else:
        return 0


def xent_1601(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
    if (weight > 0) and (obj_labels is not None):
        # hard object labels
        image_target = obj_labels
        loss = nn.CrossEntropyLoss(reduction='none')(prediction_scores_v.reshape(-1, 1601), image_target.view(-1,))
        return weight * torch.sum(loss * (label.view(-1) == 1)) / max(torch.sum((label == 1)), 1)
    else:
        return 0


pre_vis_targets = {
    "0": 1601,
    "1": 2048,
    "2": 2048,
    "3": 1600,
    "4": 400,
    "5": 2048,
    "6": 1601
}

pre_vis_criterions = {
    "0": kl_1601,
    "1": mse_2048,
    "2": nce_2048,
    "3": xent_1600,
    "4": xent_400,
    "5": huber_2048,
    "6": xent_1601,
}

# ==================================================================================================================== #
#                                                  Fine-tuning                                                  #
# ==================================================================================================================== #


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, vil_prediction, target, attn_mask_v, temperature=1.0):
        """
        :param vil_prediction: [batch, v_seq_len, 1]
        :param target: [batch, v_seq_len, 1]
        :param attn_mask_v: [batch, v_seq_len]
        :param temperature: float
        :return:
        """
        assert temperature > 0
        vil_prediction = vil_prediction.squeeze(2)  # [batch, v_seq_len]
        target = target.squeeze(2)  # [batch, v_seq_len]
        """
        print("vil_prediction")
        print(vil_prediction.size())
        print(vil_prediction[0].detach().cpu().numpy())
        print("target")
        print(target.size())
        print(target[0].detach().cpu().numpy())
        """
        log_softmax_density_all = masked_log_softmax(vil_prediction/temperature, attn_mask_v, dim=1)  # [batch, v_seq_len]
        ground_truth_region_mask = (target > 0.5).float()
        log_softmax_density_positive_samples = log_softmax_density_all * ground_truth_region_mask
        nce_loss = -log_softmax_density_positive_samples.sum(dim=1) / (ground_truth_region_mask.sum(dim=1) + 1e-13)  # [batch], add + 1e-13 to avoid NaN when there is no target
        """
        print("_prediction_log_softmax")
        print(_prediction_log_softmax[0])
        print("selection_mask")
        print(selection_mask.size())
        print(selection_mask.detach().cpu().numpy())
        print("_prediction_log_softmax")
        print(_prediction_log_softmax.size())
        print(_prediction_log_softmax.detach().cpu().numpy())
        print("nce_loss")
        print(nce_loss.size())
        print(nce_loss.detach().cpu().numpy())
        """
        return nce_loss.mean(dim=0)  #


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, vil_prediction, target, attn_mask_v, temperature=1.0):
        """
        :param vil_prediction: [batch, v_seq_len, 1]
        :param target: [batch, v_seq_len, 1]
        :param attn_mask_v: [batch, v_seq_len]
        :param temperature: float
        :return:
        """
        assert temperature > 0
        vil_prediction = vil_prediction.squeeze(2)  # [batch, v_seq_len]
        target = target.squeeze(2)  # [batch, v_seq_len]
        log_softmax_density_all = masked_log_softmax(vil_prediction/temperature, attn_mask_v, dim=1)  # [batch, v_seq_len]
        ground_truth_softmax_density_all = masked_softmax(target/temperature, attn_mask_v, dim=1)  # [batch, v_seq_len]
        loss = -torch.sum(ground_truth_softmax_density_all * log_softmax_density_all, dim=1)
        #ground_truth_region_mask = (target > 0.5).float()
        #log_softmax_density_positive_samples = log_softmax_density_all * ground_truth_region_mask
        #nce_loss = -log_softmax_density_positive_samples.sum(dim=1) / (ground_truth_region_mask.sum(dim=1) + 1e-13)  # [batch], add + 1e-13 to avoid NaN when there is no target
        return loss.mean(dim=0)  #
