#!/usr/bin/python3.7
# -*- coding:utf-8 -*-
"""
@Time : 2022/6/25 16:16
@Author: Bryan
@Comment : 相当于项目的主文件，定义了整体的训练过程
"""
import argparse
import torch
import os

# 定义了所需要文件的位置
parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default=r'D:\DATA\NLP\Models\bert-base-chinese')
parser.add_argument('--save_path', help='path to save checkpoints', default=r"D:\DATA\NLP\project\train_checkpoint")
parser.add_argument('--train_file', help='training data', default=r'D:\DATA\NLP\project\NLU\data\train.tsv')
parser.add_argument('--valid_file', help='valid data', default=r'D:\DATA\NLP\project\NLU\data\test.tsv')
parser.add_argument('--intent_label_vocab', help='training file', default=r'D:\DATA\NLP\project\NLU\data\cls_vocab')
parser.add_argument('--slot_label_vocab', help='training file', default=r'D:\DATA\NLP\project\NLU\data\slot_vocab')
# 定义了训练的超参数
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=30)
parser.add_argument('--batch_split', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # 有多个gpu，用这个方式来指定

# 下面还有import各种包而不是放在最上面，是因为如果下面的import放到最上面，上面这一句就失效了
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 就失效了
from transformers import BertConfig, BertTokenizer, AdamW
from NLU_model import NLUModule
import dataset
import utils
import traceback
from trainer import Trainer
from torch.nn.parallel import DistributedDataParallel
print("gpu use: %s " % torch.cuda.current_device())
print("gpu available: %s " % torch.cuda.is_available())

# 配置初始化目录
train_path = os.path.join(args.save_path, 'train')
log_path = os.path.join(args.save_path, 'log')

def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_path, filename)) # 只保存模型学习的参数
# 当没有目录就要创建目录
try:
    if args.local_rank == -1 or args.local_rank == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
    while not os.path.isdir(args.save_path):
        pass
    logger = utils.get_logger(os.path.join(args.save_path, 'train.log'))

    # Setup logging and save folder
    if args.local_rank == -1 or args.local_rank == 0:
        for path in [train_path, log_path]:
            if not os.path.isdir(path):
                logger.info('cannot find {}, mkdiring'.format(path))
                os.makedirs(path)

        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))
# 上面一直有一个参数是local_rank, 这是pytorch中一个很重要的是否并行训练的参数
# 如果 local_rank == -1， 就是非并行训练；如果local_rank == 0, 就表示其为并行训练中的主程序

# 下面79-87行是pytorch中的多机多卡的初始化操作，基本所有工作中都是一样的代码
    distributed = (args.local_rank != -1)
    if distributed:
        # Setup distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.manual_seed(args.seed)
    else:
        device = torch.device("cuda", 0)

    tokz = BertTokenizer.from_pretrained(args.bert_path) # 当到bert的tokenizer
    _, intent2index, _ = utils.load_vocab(args.intent_label_vocab) # 加载intent标签
    _, slot2index, _ = utils.load_vocab(args.slot_label_vocab) # 加载槽值标签
    train_dataset = dataset.NLUDataset([args.train_file], tokz, intent2index, slot2index, logger, max_lengths=args.max_length)
    valid_dataset = dataset.NLUDataset([args.valid_file], tokz, intent2index, slot2index, logger, max_lengths=args.max_length)

    logger.info('Building models, rank {}'.format(args.local_rank))
    bert_config = BertConfig.from_pretrained(args.bert_path)
    bert_config.num_intent_labels = len(intent2index) # 引入intent用到的参数长度
    bert_config.num_slot_labels = len(slot2index) # 引入槽值对应的参数长度
    model = NLUModule.from_pretrained(args.bert_path, config=bert_config).to(device) # 初始化model

    if distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank) # 如果是并行训练就用这个DDP的模块，具体的去看pytorch的DDP

    trainer = Trainer(args, model, tokz, train_dataset, valid_dataset, log_path, logger, device, distributed=distributed) # 把初始化的model，dataset， config都放在这个Trainer的类
    # 日后如果更新model的话，直接更新model中的参数就好了；并不需要动这个trainer

    start_epoch = 0
    if args.local_rank in [-1, 0]:
        trainer.train(start_epoch, args.n_epochs, after_epoch_funcs=[save_func]) # 具体的训练过程调用
    else:
        trainer.train(start_epoch, args.n_epochs) # 具体的训练过程调用

except:
    logger.error(traceback.format_exc())
