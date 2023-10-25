# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer
from volta.losses import InfoNCELoss, ListNetLoss

from volta.datasets import DatasetMapTrain, DatasetMapEval

from volta.datasets._image_features_reader import ImageFeaturesH5Reader

import numpy as np
from Mymodels.blip import  init_tokenizer
import json
logger = logging.getLogger(__name__)
import os
import pickle

# 定义一个损失函数字典，用于消融实验的时候更换模型

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "InfoNCELoss": InfoNCELoss(),
    "ListNetLoss": ListNetLoss(),
    "InfoNCESequenceLabelLoss": {"region_classification": InfoNCELoss(), "sequence_labeling": nn.BCEWithLogitsLoss(reduction="mean")},
    "BCESequenceLabelLoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "sequence_labeling": nn.BCEWithLogitsLoss(reduction="mean")},
    "BCEInfoNCELoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": InfoNCELoss()},
    "ListNetInfoNCELoss": {"region_classification": ListNetLoss(), "contrastive": InfoNCELoss()},
    "BCEListNet": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": ListNetLoss()},
    "BCEInfoNCECELoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": InfoNCELoss(), "object_categorization": nn.CrossEntropyLoss()}
}


def ForwardModelsVal(config, task_cfg, device, task_id, batch, model,model_blip,criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)

    # ------------------------------------------------------------------------------------------------------------------------------------
    #       加载数据
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # 不同任务对应着不同的type 意味着不同的读取顺序方式
    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw = batch

    elif task_cfg[task_id]["type"].startswith("VL-contrast") or task_cfg[task_id]["type"].startswith("V-logit") or task_cfg[task_id]["type"] == "VL-keywordmlp":
        
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw = batch

    elif task_cfg[task_id]["type"].startswith("VL-seq-label"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw = batch

    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw = batch

    else:
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw = batch

    # target: [batch, num_regions, 1] IOU between each proposal bounding box and the ground-truth bounding box

    batch_size = features.size(0)
    output_all_encoded_layers = False
    

    # ------------------------------------------------------------------------------------------------------------------------------------
    #       加载必要组件
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # process的方式有 normal/dialog/expand/retrieval/nlvr几种

    if task_cfg[task_id]["process"] in ["dialog"]:
        raise NotImplementedError("dialog process for validation")

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
    

    if task_cfg[task_id]["type"].startswith("V-logit-fuse") or task_cfg[task_id]["type"].startswith("VL-obj-categorize-probing"):
        output_all_encoded_layers = True
    if task_cfg[task_id]["type"] == "VL-obj-categorize-probing-mask-text":
        input_mask = torch.zeros_like(question)

    # ------------------------------------------------------------------------------------------------------------------------------------
    #  模型预测
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    img_feat,text_feat,multi_feat = model_blip(imgs,txt_raw)

    if output_all_encoded_layers:
        vil_prediction, vision_prediction, linguisic_prediction, _, _, _ = model(img_feat,text_feat,multi_feat,txt_raw,imgs,question, features, spatials,task_id,
                                                                           segment_ids,input_mask, image_mask,
                                                                           output_all_encoded_layers)
    else:
        vil_prediction, vision_prediction, linguisic_prediction, _= model(txt_raw,imgs,question, features, spatials,task_id,
                                                                           segment_ids,input_mask, image_mask,
                                                                           output_all_encoded_layers)
    # ------------------------------------------------------------------------------------------------------------------------------------
    #  损失函数选择
    # ------------------------------------------------------------------------------------------------------------------------------------
    

    if task_cfg[task_id]["type"] == "VL-classifier":

    # 来计算模型的预测值 vil_prediction 与真实目标 target 之间的损失。
        loss = criterion(vil_prediction, target)
    # 在计算完原始的损失后，这一行代码将损失取平均值（mean）并乘以 target 的长度（列数）
        loss = loss.mean() * target.size(1)
    # 计算了预测值 vil_prediction 和真实目标 target 之间的得分
        batch_score = compute_score_with_logits(vil_prediction, target).sum()





    elif task_cfg[task_id]["type"] == "V-logit-fuse-self-attention" or task_cfg[task_id]["type"] == "V-logit-fuse-self-attention-vseq-mean-pooled":
        pred_scores, layer_attn_scores = vil_prediction
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-fuse-self-attention-text-vision":
        pred_scores, layer_attn_scores_v, layer_attn_scores_t, keyword_attn_scores = vil_prediction
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-fuse-text-vision":
        pred_scores, attn_scores = vil_prediction
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-seq-label":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target)
        region_classification_loss = region_classification_loss.mean() * target.size(1)
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss
        #loss = loss.mean() * target.size(1)

        _, select_idx = torch.max(region_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-contrast":
        if task_cfg[task_id]["loss"] == "InfoNCELoss" or task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        elif task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-contrast-separated":
        pred_scores, sim_scores, attn_scores = vil_prediction
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask, task_cfg[task_id]["listnet_temperature"])
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-obj-categorize-contrast":
        pred_scores, sim_scores, tgt_obj_class_scores, attn_scores = vil_prediction
        # contrastive loss
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        
        '''region classification loss'''
        if task_cfg[task_id]["loss"].startswith("BCE"):
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"].startswith("ListNet"):
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask,
                                                                            task_cfg[task_id]["listnet_temperature"])
        else:
            raise ValueError
        # tgt object categorization loss
        tgt_object_categorization_loss = criterion["object_categorization"](tgt_obj_class_scores, ref_category_id.squeeze(1))

        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss + task_cfg[task_id]["categorization_loss_weight"] * tgt_object_categorization_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize-probing"):
        # tgt object categorization loss
        ref_category_id = ref_category_id.squeeze(1)  # [batch]
        # vil_prediction: [batch, num_classes]
        loss = criterion(vil_prediction, ref_category_id)

        _, select_obj_cat_idx = torch.max(vil_prediction, dim=1)
        #select_obj_cat_idx: [batch]
        acc = torch.sum(select_obj_cat_idx == ref_category_id).item()
        batch_score = (acc, select_obj_cat_idx.tolist(), ref_category_id.tolist())
        #batch_score = torch.sum(select_obj_cat_idx == ref_category_id).item()

    elif task_cfg[task_id]["type"] == "VL-seq-label-contrast":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id]["sequence_loss_weight"] * sequence_labeling_loss

        #loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print(loss)
        #exit()
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(region_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    #return float(loss), float(batch_score), batch_size
    return float(loss), batch_score, batch_size

 
 # 训练时候的前项传播函数
def ForwardModelsTrain(config, task_cfg, device, task_id, batch, model, criterion,model_blip):
    batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)





    # ------------------------------------------------------------------------------------------------------------------------------------
    #       加载数据
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # 不同任务对应着不同的type 意味着不同的读取顺序方式
    if task_cfg[task_id]["type"] == "V-logit-mc":
        # 
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-contrast") or task_cfg[task_id]["type"].startswith("V-logit") or task_cfg[task_id][
            "type"] == "VL-keywordmlp":
    # 实际上是这个
    # ReferExpressionDataset： features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw,class_emotion = batch
    elif task_cfg[task_id]["type"].startswith("VL-seq-label"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, sequence_labels_target = batch
    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize"):
        # 
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, ref_category_id = batch
        
        # ref_category_id: [batch, 1]
        #print("ref_category_id")
        #print(ref_category_id.size())
        #print(ref_category_id.detach().cpu().numpy())
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id= batch



    '''
    print(f"Feedforward-image_id {features.shape}")
    print(f"Feedforward-target {target.shape}")
    print(f"Feedforward-input_mask {input_mask.shape}")
    print(f"Feedforward-image_mask {image_mask.shape}")
    print(f"Feedforward-question {question.shape}")
    print(f"Feedforward-segment_ids {segment_ids.shape}")
    print(f"Feedforward-spatials {spatials.shape}")
    print(f"Feedforward-question_id {question_id.shape}")
    '''
    batch_size = features.size(0)
    output_all_encoded_layers = False
    
    
    # ------------------------------------------------------------------------------------------------------------------------------------
    #       加载必要组件
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # process的方式有 normal/dialog/expand/retrieval/nlvr几种
    # print(task_cfg[task_id]["process"]) normal

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
        segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

   
    # ------------------------------------------------------------------------------------------------------------------------------------
    #  其他的一些设置
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    if task_cfg[task_id]["type"].startswith("V-logit-fuse") or task_cfg[task_id]["type"].startswith("VL-obj-categorize-probing"):
        output_all_encoded_layers = True

    if task_cfg[task_id]["type"] == "VL-obj-categorize-probing-mask-text":
        input_mask = torch.zeros_like(question)
    

    # ------------------------------------------------------------------------------------------------------------------------------------
    #  模型预测
        '''
        input_txt, #输入的文本向量,input_imgs, #输入的图片向量,image_loc, # 输入的图片位置,task_id, # 任务id
        token_type_ids=None, # token编码的id表
        attention_mask=None, 
        image_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
        '''
    # ------------------------------------------------------------------------------------------------------------------------------------
    '''
     print(input_mask.shape)
    print(image_mask.shape)
    torch.Size([16, 80])
    torch.Size([16, 37])
    '''

    img_feat,text_feat,multi_feat = model_blip(imgs,txt_raw)
    
    if output_all_encoded_layers:
        vil_prediction, vision_prediction, linguisic_prediction, _, _, _ = model(img_feat,text_feat,multi_feat,txt_raw,imgs,question, features, spatials,task_id,
                                                                           segment_ids,input_mask, image_mask,
                                                                           output_all_encoded_layers)
    else:
        vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask, output_all_encoded_layers,imgs=imgs)

    # len(all_attention_mask)==25
    # print(type(all_attention_mask)) tuple
    

   #  all_attention_mask[0] 25
    

    
    
   # name = "volta_attation.json"
   # path = os.path.join("/home/tam/Documents/RSDLayerAttn/RSDLayerAttn","attention_result")
   # t_attention_mask = all_attention_mask[0]
   # v_attention_mask = all_attention_mask[1]

    # print(t_attention_mask[1].keys()) dict_keys(['intra_attn', 'inter_attn', 'queries', 'keys'])
    '''
    tt_attn_sublayers=[0,2,4,6,8,10,12,14,16,18,20,22]
    tv_attn_sublayers=[0,2,4,6,8,10,12,14,16,18,20,22]
    vt_attn_sublayers=[0,2,4,6,8,10,12,14,16,18,20,22]
    vv_attn_sublayers=[0,2,4,6,8,10,12,14,16,18,20,22]
     

    attn_sublayers = set(tt_attn_sublayers + tv_attn_sublayers + vt_attn_sublayers + vv_attn_sublayers)

    
    for i in attn_sublayers:
        j = i+1
        intra_attn = t_attention_mask[j]['intra_attn'] 
        inter_attn = t_attention_mask[j]['inter_attn']
        #print(type(intra_attn)) <class 'torch.Tensor'>
        #print(intra_attn.shape) torch.Size([16, 12, 37, 37]) / intra_attn.size(1)  / intra_attn.size(2) / intra_attn.size(3) 
        batch = intra_attn.size(0)
        avage_t_intra_attn = intra_attn.sum().item() / batch 
        avage_t_inter_attn = inter_attn.sum().item() / batch 

        print(f"当前为第{j}层, avage_t_intra_attn={ avage_t_intra_attn},avage_t_inter_attn={avage_t_inter_attn}")
    
    '''


  #  if not os.path.exists(path):
  #      os.makedirs(path)
    
 #   json_path = os.path.join(path,name)
    

    
    #t_attention = [i.tolist() for i in all_attention_mask[0]]
    #v_attention =[j.tolist() for j in all_attention_mask[1]]

    
   # t_attention = {"t_attention":torch.tensor(all_attention_mask[0]).tolist()}
   # v_attention = {)"t_attention":torch.tensor(all_attention_mask[1]).tolist()}
    '''
    with open(json_path,"w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(attention) ) 
    '''
 #   print(f"Feedforward- vil_prediction1 {vil_prediction.shape}")
#    print(f"Feedforward- vision_prediction {vision_prediction.shape}") NoneType
   # print(f"Feedforward- linguisic_prediction {linguisic_prediction.shape}")
    #print(f"BertForVLTasks-  all_attention_mask {all_attention_mask.shape}")
    # ------------------------------------------------------------------------------------------------------------------------------------
    #  损失函数选择
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # 对于不同种类的任务使用不同的方式来计算损失函数
    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":

     # 来计算模型的预测值 vil_prediction 与真实目标 target 之间的损失。
        loss = criterion(vil_prediction, target)
    # 在计算完原始的损失后，这一行代码将损失取平均值（mean）并乘以 target 的长度（列数）
        loss = loss.mean() * target.size(1)
    # 计算了预测值 vil_prediction 和真实目标 target 之间的得分
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)


    elif task_cfg[task_id]["type"] == "V-logit-fuse-self-attention" or task_cfg[task_id]["type"] == "V-logit-fuse-self-attention-vseq-mean-pooled":
       
        pred_scores, layer_attn_scores = vil_prediction
        
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    return loss, batch_score



def LoadLoss(task_cfg, task_id):
    task = "TASK" + task_id
    loss = LossMap[task_cfg[task]["loss"]]
    return loss


def LoadDataset(args, config, task_cfg, task_id,data_imgs_root,split="trainval"):

    # 导入预处理编码器，采用bert的base模型 "bert_model": "bert-base-uncased",
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    
    task = "TASK" + task_id

    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    #   features_h5path1: datasets/flickr30k/resnet101_faster_rcnn_genome_imgfeats/volta/flickr30k_feat.lmdb
    #   features_h5path2: train_annotations_jsonpath: datasets/flickr30k/annotations/train_ann.jsonl

    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]

    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None
 
    batch_size = task_cfg[task]["batch_size"] // args.grad_acc_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    dset_train, dset_train, task2num_iters = None, None, {}
    
    # /home/tam/Documents/RSDLayerAttn/RSDLayerAttn/volta/datasets/refer_expression_dataset.py
    # "talk2car": ReferExpressionDataset
    dset_train = DatasetMapTrain[task_name](
        task=task_cfg[task]["name"],
        dataroot=task_cfg[task]["dataroot"],
        annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
        split=task_cfg[task]["train_split"],
        image_features_reader=features_reader1, # 主要是用这个
        gt_image_features_reader=features_reader2, # 这个特征好像基本没用到，config_task里面也没有
        tokenizer=tokenizer,
        bert_model=config.bert_model,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
        num_locs=config.num_locs,
        add_global_imgfeat=config.add_global_imgfeat,
        append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
    )
    
    if args.local_rank == -1:
        #train_sampler = RandomSampler(dset_train)
        if args.weighted_sampling:
            train_sampler = WeightedRandomSampler(dset_train.sample_category_weights, num_samples=len(dset_train),
                                                    replacement=True)
        else:
            train_sampler = RandomSampler(dset_train)
    else:
        train_sampler = DistributedSampler(dset_train)
        
    dl_train = DataLoader(
        dset_train,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=args.drop_last,

    )

    task2num_iters = {task: len(dl_train)}

    dset_val, dl_val = None, None
    '''
        dset_train = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=features_reader1, # 主要是用这个
            gt_image_features_reader=features_reader2, # 这个特征好像基本没用到，config_task里面也没有
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
    )
    '''

    dset_val = DatasetMapTrain[task_name](
        task=task_cfg[task]["name"],
        dataroot=task_cfg[task]["dataroot"],
        annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
        split=task_cfg[task]["val_split"],
        image_features_reader=features_reader1,
        gt_image_features_reader=features_reader2,
        tokenizer=tokenizer,
        bert_model=config.bert_model,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
        num_locs=config.num_locs,
        add_global_imgfeat=config.add_global_imgfeat,
        append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),

    )
    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        # batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=args.drop_last,
    )

    return batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val


def LoadDatasetEval(args, config, task_cfg, task_id):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    
    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    if task_name.startswith("Retrieval"):
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            num_subiters=args.num_subiters,
        )
    else:
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
       # num_workers=10,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

def compute_binary_sequence_label_score_with_logits(logits, labels):
    """
    :param logits: [batch, seq_len, 1]
    :param labels: [batch, seq_len, 1]
    :return:
    """
    predicted_prob = torch.sigmoid(logits.squeeze(2))
    prediction = (predicted_prob > 0.5).float()
    print("prediction")
    print(prediction.size())
    print(prediction[0])
    print("labels")
    print(labels[0,:,0])
    match_tensor = prediction == labels.squeeze(2)
    print("match_tensor")
    print(match_tensor.size())
    print(match_tensor[0])
    batch_score = match_tensor.sum(1).sum(0).item()
    return batch_score

import sys
def MyEvaluatingModel(config, task_cfg, device, task_id, batch, model, model_blip,dataloader, criterion, results, others, bbox):
    batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)

    features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id,imgs,txt_raw,class_emotion = batch

    batch_size = features.size(0)
    output_all_encoded_layers = False

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    if task_cfg[task_id]["type"].startswith("V-logit-fuse") or task_cfg[task_id]["type"].startswith("VL-obj-categorize-probing") or task_cfg[task_id]["type"].startswith("VL-visualization"):
        output_all_encoded_layers = True

    if task_cfg[task_id]["type"] == "VL-obj-categorize-probing-mask-text":
        input_mask = torch.zeros_like(question)
    import time
    model_inference_start_time = time.time()
    with torch.no_grad():
        
        img_feat,text_feat,multi_feat = model_blip(imgs,txt_raw)

        if output_all_encoded_layers:
            vil_prediction, vision_prediction, linguisic_prediction, _, _, _ = model(img_feat,text_feat,multi_feat,txt_raw,imgs,question, features, spatials,task_id,
                                                                            segment_ids,input_mask, image_mask,
                                                                            output_all_encoded_layers)
        else:
            vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                        segment_ids, input_mask, image_mask, output_all_encoded_layers,imgs=imgs)


    model_inference_time = time.time() - model_inference_start_time
    selection_time = 0

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-seq-label":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target)
        region_classification_loss = region_classification_loss.mean() * target.size(1)
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss

        _, select_idx = torch.max(region_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id]["type"] == "VL-keywordmlp" or task_cfg[task_id]["type"] == "V-logit-fuse" or task_cfg[task_id]["type"] == "V-logit-fuse-coarse-attention" or task_cfg[task_id]["type"] == "V-logit-fuse-fine-attention" or task_cfg[task_id]["type"] == "V-logit-fuse-dynamic" or task_cfg[task_id]["type"] == "V-logit-fuse-routing-by-agreement":
        if task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print()
        #print(vil_prediction.size())
        #print("vil_prediction")
        #print(vil_prediction[0,:,0])
        # vli_predition: [batch, num_regions, 1]
        selection_start_time = time.time()
        _, select_idx = torch.max(vil_prediction, dim=1)
        selection_time = time.time() - selection_start_time
        #print(select_idx.size())
        #print("idx")
        #print(select_idx[0])
        #print("target")
        #print(target.size())
        #print("spatials_ori")
        #print(spatials_ori[select_idx[0],:4])
        #exit()
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "region_all_IOU": target[i].tolist()
                }
            )
    elif task_cfg[task_id]["type"] == "V-logit-fuse-self-attention" or task_cfg[task_id]["type"] == "V-logit-fuse-self-attention-vseq-mean-pooled":

        pred_scores, layer_attn_scores = vil_prediction

        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        #print()
        #print(vil_prediction.size())
        #print("vil_prediction")
        #print(vil_prediction[0,:,0])
        # vli_predition: [batch, num_regions, 1]
        selection_start_time = time.time()
        _, select_idx = torch.max(pred_scores, dim=1)

        largest_indices = torch.topk(pred_scores, k=6, dim=1)
        p6_largest_indices = torch.topk(pred_scores, k=6, dim=1, largest=False)[1][:, 0]
        p5_largest_indices = torch.topk(pred_scores, k=5, dim=1, largest=False)[1][:, 0]
        p4_largest_indices = torch.topk(pred_scores, k=4, dim=1, largest=False)[1][:, 0]
        p3_largest_indices = torch.topk(pred_scores, k=3, dim=1, largest=False)[1][:, 0]
        p2_largest_indices = torch.topk(pred_scores, k=2, dim=1, largest=False)[1][:, 0]
        
        selection_time = time.time() - selection_start_time

        #print(select_idx.size())
        #print("idx")
        #print(select_idx[0])
        #print("target")
        #print(target.size())
        #print("spatials_ori")
        #print(spatials_ori[select_idx[0],:4])
        #exit()
        
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        
        batch_score = torch.sum(select_target > 0.5).item()
        layer_attn_scores = layer_attn_scores.tolist()
        
        
        image_mask = image_mask.tolist()  # [batch, v_seq_len]


        '''
        print("pred_scores",pred_scores.shape)
        print("layer_attn_scores",layer_attn_scores.shape)
        pred_scores torch.Size([256, 37, 1])

        layer_attn_scores torch.Size([256, 37, 13])
        select_idx torch.Size([256, 1])
        '''
        
        
        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "v_seq_len": sum(image_mask[i]),
                    "layer_attn_scores": layer_attn_scores[i],
                    "region_all_IOU": target[i].tolist()
                }
            )

    elif task_cfg[task_id]["type"] == "VL-seq-label-contrast":
        #loss = criterion(vil_prediction, target)

        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target, image_mask,
                                                                        task_cfg[task_id]["temperature"])
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss

        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(region_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "VL-contrast":
        if task_cfg[task_id]["loss"] == "InfoNCELoss" or task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        elif task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )
    elif task_cfg[task_id]["type"] == "VL-contrast-separated":
        pred_scores, sim_scores, attn_scores = vil_prediction
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask,
                                                   task_cfg[task_id]["listnet_temperature"])
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        attn_scores = attn_scores.detach().cpu().tolist()
        query_list = question.tolist()  # [batch, seq_len]

        # debug
        """
        for i in range(pred_scores.size(0)):
            print("target")
            print(target[i].detach().squeeze(1).cpu().numpy())
            print("pred_scores")
            print(pred_scores[i].detach().squeeze(1).cpu().numpy())
            print("sim_scores")
            print(sim_scores[i].detach().squeeze(1).cpu().numpy())
            print("region_classification_loss")
            print(region_classification_loss.item())
            print("contrastive_loss")
            print(contrastive_loss.item())
            print()
        exit()
        """

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "attention_score": attn_scores[i],
                    "query": dataloader.dataset._tokenizer.convert_ids_to_tokens(query_list[i])
                }
            )

    elif task_cfg[task_id]["type"] == "VL-obj-categorize-contrast":
        pred_scores, sim_scores, tgt_obj_class_scores, attn_scores = vil_prediction
        # contrastive loss
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        # region classification loss
        if task_cfg[task_id]["loss"].startswith("BCE"):
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"].startswith("ListNet"):
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask,
                                                                            task_cfg[task_id]["listnet_temperature"])
        else:
            raise ValueError
        # tgt object categorization loss
        tgt_object_categorization_loss = criterion["object_categorization"](tgt_obj_class_scores, ref_category_id.squeeze(1))

        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss + task_cfg[task_id]["categorization_loss_weight"] * tgt_object_categorization_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        attn_scores = attn_scores.detach().cpu().tolist()

        # debug
        """
        for i in range(pred_scores.size(0)):
            print("target")
            print(target[i].detach().squeeze(1).cpu().numpy())
            print("pred_scores")
            print(pred_scores[i].detach().squeeze(1).cpu().numpy())
            print("sim_scores")
            print(sim_scores[i].detach().squeeze(1).cpu().numpy())
            print("region_classification_loss")
            print(region_classification_loss.item())
            print("contrastive_loss")
            print(contrastive_loss.item())
            print()
        exit()
        """

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i], :4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
                {
                    question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2] - bbox_item[0],
                                            bbox_item[3] - bbox_item[1]]
                }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "attention_score": attn_scores[i]
                }
            )
    elif task_cfg[task_id]["type"] == "VL-visualization":
        pred_scores, layer_attn_scores, sequence_output_v_sample, image_attention_mask, fused_representation_v = vil_prediction
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        # print()
        # print(vil_prediction.size())
        # print("vil_prediction")
        # print(vil_prediction[0,:,0])
        # vli_predition: [batch, num_regions, 1]
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx.size())
        # print("idx")
        # print(select_idx[0])
        # print("target")
        # print(target.size())
        # print("spatials_ori")
        # print(spatials_ori[select_idx[0],:4])
        # exit()
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        layer_attn_scores = layer_attn_scores.tolist()
        image_mask = image_mask.tolist()  # [batch, v_seq_len]

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i], :4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
                {
                    question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2] - bbox_item[0],
                                            bbox_item[3] - bbox_item[1]]
                    # question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
                }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "v_seq_len": sum(image_mask[i]),
                    "layer_attn_scores": layer_attn_scores[i],
                    "region_all_IOU": target[i].tolist()
                }
            )

        # Dump sequence_output_v_sample, image_attention_mask
        # list with len=13
        sequence_output_v_sample_cpu = []
        for layer in sequence_output_v_sample:
            sequence_output_v_sample_cpu.append(layer.cpu())
        image_attention_mask = image_attention_mask.cpu()  # [batch_size, v_seq_len]
        #torch.save(sequence_output_v_sample_cpu, './visualization_output/lxmert_sequence_output_v.pt')
        #torch.save(image_attention_mask, './visualization_output/lxmert_image_attention_mask.pt')
        torch.save(fused_representation_v, './visualization_output/lxmert_fused_representation_v.pt')
        print("sequence_output_v_sample_cpu")
        print(len(sequence_output_v_sample_cpu))
        print(sequence_output_v_sample_cpu[0].size())
        print("image_attention_mask")
        print(image_attention_mask.size())
        print("fused_representation_v")
        print(fused_representation_v.size())
        print("finished")
        exit()

    elif task_cfg[task_id]["type"] == "VL-visualization-original":
        pred_scores, sequence_output_v_sample, image_attention_mask = vil_prediction
        loss = criterion(pred_scores, target)
        loss = loss.mean() * target.size(1)
        # print()
        # print(vil_prediction.size())
        # print("vil_prediction")
        # print(vil_prediction[0,:,0])
        # vli_predition: [batch, num_regions, 1]
        _, select_idx = torch.max(pred_scores, dim=1)
        # print(select_idx.size())
        # print("idx")
        # print(select_idx[0])
        # print("target")
        # print(target.size())
        # print("spatials_ori")
        # print(spatials_ori[select_idx[0],:4])
        # exit()
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        #layer_attn_scores = layer_attn_scores.tolist()
        image_mask = image_mask.tolist()  # [batch, v_seq_len]

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i], :4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
                {
                    question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2] - bbox_item[0],
                                            bbox_item[3] - bbox_item[1]]
                    # question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
                }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "v_seq_len": sum(image_mask[i]),
                    #"layer_attn_scores": layer_attn_scores[i],
                    "region_all_IOU": target[i].tolist()
                }
            )

        # Dump sequence_output_v_sample, image_attention_mask
        # list with len=13
        sequence_output_v_sample_cpu = []
        for layer in sequence_output_v_sample:
            sequence_output_v_sample_cpu.append(layer.cpu())
        image_attention_mask = image_attention_mask.cpu()  # [batch_size, v_seq_len]
        torch.save(sequence_output_v_sample_cpu, './visualization_output/uniter_sequence_output_v.pt')
        torch.save(image_attention_mask, './visualization_output/uniter_image_attention_mask.pt')
        #torch.save(fused_representation_v, './visualization_output/lxmert_fused_representation_v.pt')
        print("sequence_output_v_sample_cpu")
        print(len(sequence_output_v_sample_cpu))
        print(sequence_output_v_sample_cpu[0].size())
        print("image_attention_mask")
        print(image_attention_mask.size())
        #print("fused_representation_v")
        #print(fused_representation_v.size())
        print("finished")
        exit()

    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize-probing"):
        # tgt object categorization loss
        ref_category_id = ref_category_id.squeeze(1)  # [batch]
        # vil_prediction: [batch, num_classes]
        loss = criterion(vil_prediction, ref_category_id)

        _, select_obj_cat_idx = torch.max(vil_prediction, dim=1)

        # random guess
        #batch_size, num_classes = vil_prediction.size()
        #select_obj_cat_idx = torch.randint(low=0, high=num_classes, size=(batch_size,1)).squeeze(1).cuda()
        # always guess vehicle.car
        #batch_size, num_classes = vil_prediction.size()
        #select_obj_cat_idx = torch.ones(batch_size).cuda() * 3

        # select_obj_cat_idx: [batch]
        acc = torch.sum(select_obj_cat_idx == ref_category_id).item()
        batch_score = (acc, select_obj_cat_idx.tolist(), ref_category_id.tolist())
        # batch_score = torch.sum(select_obj_cat_idx == ref_category_id).item()

        for i in range(select_obj_cat_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "prediction": select_obj_cat_idx[i].item(),
                    "target": ref_category_id[i].item()
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    #return float(loss), float(batch_score), batch_size, results, others, bbox

    return float(loss), batch_score, batch_size, results, others, bbox, model_inference_time+selection_time




