# imports
import os
import argparse
import json
import shutil
import sys
import warnings
sys.path.append("Talk2Car")
sys.path.append("/Talk2Car/baseline")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

# from utils.collate import custom_collate
# from utils.util import AverageMeter, ProgressMeter, save_checkpoint

#import models.resnet as resnet
#import models.nlp_models as nlp_models
import parser
import sys
sys.path.append("/home/tam/Documents/RSDLayerAttn/MyTrain/Models/utils/dataloader/")
from Models.utils.dataloader.talk2car import Talk2Car

def main(args):

    # Create dataset
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    train_dataset = Talk2Car(talk2car_root=args.root, split='train', bbox_file="/content/Talk2Car/baseline/data/centernet_bboxes.json",
                             vocabulary="/content/Talk2Car/baseline/utils/vocabulary.txt",
                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
    val_dataset = Talk2Car(talk2car_root=args.root, split='val',bbox_file="/content/Talk2Car/baseline/data/centernet_bboxes.json",
                           vocabulary="/content/Talk2Car/baseline/utils/vocabulary.txt",
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,drop_last=False)

    # Create model
    print("=> creating model")
    img_encoder = resnet.__dict__['resnet18'](pretrained=True) 
    text_encoder = nlp_models.TextEncoder(input_dim=train_dataset.number_of_words(),
                                                 hidden_size=512, dropout=0.1)
    img_encoder.cuda()
    text_encoder.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index = train_dataset.ignore_index, 
                                    reduction = 'mean')
    criterion.cuda()    
    
    cudnn.benchmark = True

    # Optimizer and scheduler
    print("=> creating optimizer and scheduler")
    params = list(text_encoder.parameters()) + list(img_encoder.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                            gamma=0.1)

    # Checkpoint
    checkpoint = 'checkpoint.pth.tar'
    if os.path.exists(checkpoint):
        print("=> resume from checkpoint at %s" %(checkpoint))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_ap50 = checkpoint['best_ap50']
    else:
        print("=> no checkpoint at %s" %(checkpoint))
        best_ap50 = 0
        start_epoch = 0

    # Start training
    print("=> start training")

    for epoch in range(start_epoch, args.epochs):
        print('Start epoch %d/%d' %(epoch, args.epochs))
        print(20*'-')

        # Train 
        train(train_dataloader, img_encoder, text_encoder, optimizer, criterion, epoch, args)
        
        # Update lr rate
        scheduler.step()
        
        # Evaluate
        ap50 = evaluate(val_dataloader, img_encoder, text_encoder, args)
        
        # Checkpoint
        if ap50 > best_ap50:
            new_best = True
            best_ap50 = ap50
        else:
            new_best = False

        save_checkpoint({'img_encoder': img_encoder.state_dict(),
                         'text_encoder': text_encoder.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'epoch': epoch + 1, 'best_ap50': best_ap50}, new_best = new_best)

    # Evaluate
    if args.evaluate:
        print("=> Evaluating best model")
        checkpoint = torch.load('best_model.pth.tar', map_location='cpu')
        img_encoder.load_state_dict(checkpoint['img_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        ap50 = evaluate(val_dataloader, img_encoder, text_encoder, args)
        print('AP50 on validation set is %.2f' %(ap50*100))



def train(train_dataloader, img_encoder, text_encoder, optimizer, criterion,epoch, args):
    m_losses = AverageMeter('Loss', ':.4e')
    m_top1 = AverageMeter('Acc@1', ':6.2f')
    m_iou = AverageMeter('IoU', ':6.2f')
    m_ap50 = AverageMeter('AP50', ':6.2f')
    progress = ProgressMeter(
                len(train_dataloader),
                [m_losses, m_top1, m_iou, m_ap50], prefix="Epoch: [{}]".format(epoch))
 
    img_encoder.train()
    text_encoder.train()
    
    ignore_index = train_dataloader.dataset.ignore_index
        
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)
        gt = batch['rpn_gt'].cuda(non_blocking=True)
        iou = batch['rpn_iou'].cuda(non_blocking=True)
        b, r, c, h, w = region_proposals.size()

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm)
       
        # Sentence features
        _, sentence_features = text_encoder(command.permute(1,0), command_length)
        norm = sentence_features.norm(p=2, dim=1, keepdim=True)
        sentence_features = sentence_features.div(norm)
     
        # Product in latent space
        scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()

        # Loss
        total_loss = criterion(scores, gt)

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Summary
        pred = torch.argmax(scores, 1)
        pred_bin = F.one_hot(pred, r).bool()
        valid = (gt!=ignore_index)
        num_valid = torch.sum(valid).float().item()
        m_top1.update(torch.sum(pred[valid]==gt[valid]).float().item(), num_valid)
        m_iou.update(torch.masked_select(iou, pred_bin).sum().float().item(), b)
        m_ap50.update((torch.masked_select(iou, pred_bin) > 0.5).sum().float().item(), b)
        m_losses.update(total_loss.item())

        if i % args.print_freq==0:
            progress.display(i)


@torch.no_grad()
def evaluate(val_dataloader, img_encoder, text_encoder, args):
    m_top1 = AverageMeter('Acc@1', ':6.2f')
    m_iou = AverageMeter('IoU', ':6.2f')
    m_ap50 = AverageMeter('AP50', ':6.2f')
    progress = ProgressMeter(
                len(val_dataloader),
                [m_top1, m_iou, m_ap50], 
                prefix='Test: ')
 
    img_encoder.eval()
    text_encoder.eval()
    
    ignore_index = val_dataloader.dataset.ignore_index
 
    for i, batch in enumerate(val_dataloader):
        
        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)
        b, r, c, h, w = region_proposals.size()

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm)
       
        # Sentence features
        _, sentence_features = text_encoder(command.permute(1,0), command_length)
        norm = sentence_features.norm(p=2, dim=1, keepdim=True)
        sentence_features = sentence_features.div(norm)
     
        # Product in latent space
        scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()

        # Summary
        pred = torch.argmax(scores, 1)
        gt = batch['rpn_gt'].cuda(non_blocking=True)
        iou = batch['rpn_iou'].cuda(non_blocking=True)
        pred_bin = F.one_hot(pred, r).bool()
        valid = (gt!=ignore_index)
        num_valid = torch.sum(valid).float().item()
        m_top1.update(torch.sum(pred[valid]==gt[valid]).float().item(), num_valid)
        m_iou.update(torch.masked_select(iou, pred_bin).sum().float().item(), b)
        m_ap50.update((torch.masked_select(iou, pred_bin) > 0.5).sum().float().item(), b)
        if i % args.print_freq==0:
            progress.display(i)

    return m_ap50.avg   


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--MEconfig", default="/home/tam/Documents/RSDLayerAttn/RSDLayerAttn/Mymodels/med_config.json", type=str,
                    help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--probe_layer_idx", default=None, type=int,
                        help="The layer to probe for layer probing")
    parser.add_argument("--weighted_sampling", default=False, action='store_true',
                        help="Use weighted random sampler for imbalanced learning.")
    # Training
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Scheduler
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")
                        
    parser.add_argument("--blip_pretrained", default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth', type=str,
                        )

    return parser.parse_args()



if __name__ == '__main__':
    # Run the code
    torch.manual_seed(3407)
    args = parse_args()
    # Set some args such that we are actually calling 
    # python3 train.py --root ./data --lr 0.01 --nesterov --evaluate 

    args.root = "./Talk2Car/data"
    args.lr = 0.01
    args.nesterov = True
    args.evaluate = True
    args.train_batch_size = 16
    args.test_batch_size = 32
    args.workers=8



    args.image_root = "home/tam/Documents/RSDLayerAttn/MyTrain/Talk2Car/data/image"
    args.vocabulary_root = "/home/tam/Documents/RSDLayerAttn/MyTrain/Models/utils/dataloader/vocabulary.txt"
    split = "train"

    data_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((350,350)),
        # transforms.CenterCrop(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    trian_dataset = Talk2Car(args.image_root,"train",args.vocabulary_root,transform=data_transforms)
    # val_dataset = Talk2Car(args.image_root,"val",args.vocabulary_root, transforms.ToTensor())
    # test_dataset = Talk2Car(args.image_root,"test",args.vocabulary_root, transforms.ToTensor())

    train_dataloader = data.DataLoader(trian_dataset, batch_size = args.train_batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True,drop_last=True)
    
    # val_dataloader = data.DataLoader(val_dataset, batch_size = args.test_batch_size, shuffle=False,
    #                         num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,drop_last=False)
    
    # val_dataloader = data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False,
    #                         num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,drop_last=False)




    for i, batch in enumerate(train_dataloader):
        # print(batch.keys()) index', 'rpn_bbox_lbrt', 'rpn_name_lbrt', 'rpn_score_lbrt', 'orig_image', 'image', 'orig_phrase', 'phrase', 'phrase_mask', 'gt_bbox_lbrt'
        print(len(batch["gt_bbox"]))
        print(batch["gt_bbox"][1].shape)

        
        sys.exit(1)
    
