#!/usr/bin/python3.7
# -*- coding:utf-8 -*-
"""
@Time : 2022/6/25 16:16
@Author: Bryan
@Comment : NLUModule.forward函数, 这⼀函数定义了NLU模型的计算流程(虽然⼤部分的计算是在调⽤self.bert时实现的)。
"""
from transformers import BertPreTrainedModel, BertModel
from torch import nn


class NLUModule(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intent_labels = config.num_intent_labels
        self.num_slot_labels = config.num_slot_labels

        self.bert = BertModel(config) # 得先有一个带有config初始化的Bert模型，这个Bert是来自于transformers的
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent_labels) # 这是意图识别分类器
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot_labels) # 这是槽值检测分类器
        # 这两个都是一个MLP，线性分类头，因为Bert已经足够强大了，只需要一个线性分类头，我们认为就能得到很好的性能
        # 初始化的时候需要intent和slot的数量
        # 比如intent的数量是10，它会给出10个实数；这10个实数在过完了softmax之后，就会得到10个意图的概率分布

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids, # 这里把句子的输入表示喂到bert中
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        pooled_output = outputs[1] # 这个输出的就是cls的token，代表整个句子的表示
        seq_encoding = outputs[0] # 这个输出的就是序列的token，但是肯定是batch训练，所以其长度就是batch size的长度

        pooled_output = self.dropout(pooled_output) # 这里cls过一个dropout
        intent_logits = self.intent_classifier(pooled_output) # 然后再喂到意图识别分类器中
        slot_logits = self.slot_classifier(seq_encoding) # 再把句子的序列表示喂到槽值分类器
        return intent_logits, slot_logits # logits of intent and slot classification
