# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict
from sklearn.metrics import f1_score

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

from volta.config import BertConfig
from RSDLayerAttn.volta.encoders_v0 import MyBertForVLTasks
from volta.train_utils import tbLogger
from volta.task_utils import LoadDatasetEval, LoadLoss, MyEvaluatingModel
from Mymodels.blip_itm import blip_itm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    
    parser.add_argument("--config_file", default="config/bert_config.json", type=str,
                        help="The config file which specified the model details.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--probe_layer_idx", default=None, type=int,
                        help="The layer to probe for layer probing")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--batch_size", default=30, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--blip_pretrained", default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth', type=str,
                        )

    return parser.parse_args()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")

def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    print("task id")
    print(task_id)

    # Output dirs
    timeStamp = args.from_pretrained.split("/")[-1] + "-" + args.save_name
    
    savePath = os.path.join(args.output_dir)
    print('-'*180)
    print(savePath)
    print('-'*180)
    
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config, task_cfg, args.task)

    # Logging
    tb_logger = tbLogger(timeStamp, savePath, [task_name], [task], task2num_iters,
                         1, save_logger=False, txt_name="eval.txt")



    
    # sys.exit(1)

    model = MyBertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task], probe_layer_idx=args.probe_layer_idx)
    model_blip = blip_itm(pretrained=args.blip_pretrained)
        
    # Optimization details
    criterion = LoadLoss(task_cfg, args.task)

    # Move to GPU(s)
    model.to(device)
    model_blip.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    bbox = []

    if task_id == "91" or task_id == "95":  # for computing micro f1 score for probing task
        pred_all = []
        ref_all = []
        print("init holder for f1 score")

    total_inference_time = 0

    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        
        loss, score, batch_size, results, others, bbox, inference_time = MyEvaluatingModel(config, task_cfg, device, task, batch,
                                                                   model, model_blip,dl_val, criterion, results, others, bbox)

        total_inference_time += inference_time
        if task_id == "91" or task_id == "95":  # for probing task
            acc_score, pred_list, ref_list = score
            pred_all += pred_list
            ref_all += ref_list
            score = acc_score
        tb_logger.step_val(0, float(loss), float(score), task, batch_size, "val")
        sys.stdout.write("%d/%d\r" % (i, len(dl_val)))
        sys.stdout.flush()
    # save the result or evaluate the result.
    if task_id == "91" or task_id == "95":
        acc_score = tb_logger.showLossVal(task)
        micro_f1 = f1_score(ref_all, pred_all, average='micro')
        macro_f1 = f1_score(ref_all, pred_all, average='macro')
        print("acc: {:.2f}, micro f1: {:.2f}, macro f1: {:.2f}".format(acc_score*100, micro_f1*100, macro_f1*100))
        print("pred_all")
        print(pred_all)
        print("ref_all")
        print(ref_all)
    else:
        ave_score = tb_logger.showLossVal(task)

    avg_inference_time = total_inference_time / len(dl_val)
    print("Total inference time: {:.5f}".format(total_inference_time))
    print("Inference time per batch: {:.5f}".format(avg_inference_time))

    if args.split:
        json_path = os.path.join(savePath, args.split)
    else:
        json_path = os.path.join(savePath, task_cfg[task]["val_split"])
        
    json.dump(results, open(json_path + "_result.json", "w"))
    json.dump(others, open(json_path + "_others.json", "w"))
    json.dump(bbox, open(json_path + "_prediction.json", "w"))


if __name__ == "__main__":
    main()
