#!/usr/bin/python3.7
# -*- coding:utf-8 -*-
"""
@Time : 2022/9/4 12:13
@Author: Bryan
@Comment : None
"""
import torch
from NLU_model import NLUModule

model_path = r'D:\DATA\NLP\简历\综合提升\NLP必备\对话系统模块\project\对话系统课程项目\train_checkpoint\train\model-29.ckpt'
model = NLUModule()
model.load_state_dict(torch.load(model_path))
model.eval()

model.er

