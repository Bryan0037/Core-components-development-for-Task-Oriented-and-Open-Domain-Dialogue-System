from torch.utils.data import Dataset
import torch


class NLUDataset(Dataset):
    def __init__(self, paths, tokz, cls_vocab, slot_vocab, logger, max_lengths=2048):
        self.logger = logger
        self.data = NLUDataset.make_dataset(self, paths, tokz, cls_vocab, slot_vocab, logger, max_lengths)
        # 首先这里调用的时候，也需要加self

    # @staticmethod
    # def make_dataset(paths, tokz, cls_vocab, slot_vocab, logger, max_lengths=2048):
    #     logger.info('reading data from {}'.format(paths))
    #     dataset = []
    #     for path in paths: # path 应该是字符串， paths 是list
    #         with open(path, 'r', encoding='utf8') as f:
    #             # f.readlines() 是所有的行，并且有分隔符
    #             lines = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    #             # i.strip()是去掉\n
    #             lines = [i.split('\t') for i in lines]
    #             # 此时的是每一条一个list，并且内部用,分割
    #             for label, utt, slots in lines:
    #                 utt = tokz.convert_tokens_to_ids(list(utt)[:max_lengths])
    #                 # print(slots)
    #                 slots = [slot_vocab[i] for i in slots.split()]
    #                 assert len(utt) == len(slots)
    #                 dataset.append([int(cls_vocab[label]),
    #                                 [tokz.cls_token_id] + utt + [tokz.sep_token_id],
    #                                 tokz.create_token_type_ids_from_sequences(token_ids_0=utt),
    #                                 [tokz.pad_token_id] + slots + [tokz.pad_token_id]])
    #     # print(dataset)
    #
    #     logger.info('{} data record loaded'.format(len(dataset)))
    #     return dataset

    # 原先的 @staticmethod 会导致__getitem__失效，需要让该类是普通类在class中
    # @staticmethod 和 @ classmethod 是两种特殊的方法
    # @classmethod的方法，可以通过实例或者类的方法调用，这时会把类传给第一个参数cls
    # @staticmethod 方法，不用传第一个参数
    def make_dataset(self, paths, tokz, cls_vocab, slot_vocab, logger, max_lengths=2048):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:  # path 应该是字符串， paths 是list
            with open(path, 'r', encoding = 'utf8') as f:
                # f.readlines() 是所有的行，并且有分隔符
                lines = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
                # i.strip()是去掉\n
                lines = [i.split('\t') for i in lines]
                # 此时的是每一条一个list，并且内部用,分割
                for label, utt, slots in lines:
                    utt = tokz.convert_tokens_to_ids(list(utt)[:max_lengths])
                    # print(slots)
                    slots = [slot_vocab[i] for i in slots.split()]
                    assert len(utt) == len(slots)
                    dataset.append([int(cls_vocab[label]),
                                    [tokz.cls_token_id] + utt + [tokz.sep_token_id],
                                    tokz.create_token_type_ids_from_sequences(token_ids_0 = utt),
                                    [tokz.pad_token_id] + slots + [tokz.pad_token_id]])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # 当用idx索引来取值时，回自动调用下列方法，所以list就变成了一个dictionary
        intent, utt, token_type, slot = self.data[idx]
        # print('getitem has been used')
        # return intent
        return {"intent": intent, "utt": utt, "token_type": token_type, "slot": slot}


class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        # torch.tensor 是一种包含单一数据类型的多维矩阵
        # tensor 有cpu tensor， 还有gpu tensor， torch.tensor 默认是torch.FloatTensor
        # data type :64-bit integer(signed) 对应的是torch.LongTensor
        # 64代表用64个位数二进制来表示，signed or unsigned 表示首位是否表示正负，如果可以保证正整数，可以用unsigned
        # 如果是unsigned就是2*64次方的正整数，如果是signed就是[-2*63次方， 2*63次方]
        res['intent'] = torch.LongTensor([i['intent'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([[1] * len(i['utt']) + [0] * (max_len - len(i['utt'])) for i in batch])
        res['token_type'] = torch.LongTensor(
            [i['token_type'] + [self.pad_id] * (max_len - len(i['token_type'])) for i in batch])
        res['slot'] = torch.LongTensor([i['slot'] + [self.pad_id] * (max_len - len(i['slot'])) for i in batch])
        # print(res)
        # 这里加了一个mask 的标记，所有的词的mask_id 都是1
        # 这里是按照当前batch最长的utterance而做了一个补全，这样每个batch的长度是不一样的，不会影响吗？
        # Question: """What is the advantage to applying padding on a model's batch rather than the entire dataset?
        # Confused here with this one when it comes to Recurrent Neural Networks..."""

        # Answer: """The sequences in your dataset have different lengths.
        # If you pad based on each batch, you only have to pad to match the longest length in the batch.
        # But if you pad the entire dataset, you have to make every sequence as long as the longest in the dataset.
        # This becomes wasteful: You now have to run your RNN over a lot more data that's just padding, rather than actual content."""

        return PinnedBatch(res)


if __name__ == '__main__':
    from transformers import BertTokenizer, AutoTokenizer
    bert_path = r'D:\DATA\NLP\Models\bert-base-chinese'
    # AutoTokenizer.from_pretrained("bert-base-chinese") 网好的话就可以
    # bert_path = r'/home/data/tmp/bert-base-chinese'
    data_file = r'D:\DATA\NLP\project\NLU\data\train.tsv'
    cls_vocab_file = r'D:\DATA\NLP\project\NLU\data\cls_vocab'
    slot_vocab_file = r'D:\DATA\NLP\project\NLU\data\slot_vocab'
    with open(cls_vocab_file) as f:
        res = [i.lower().strip() for i in f.readlines() if len(i.strip()) != 0]
    cls_vocab = dict(zip(res, range(1, len(res)+1)))
    # cls_vocab示例 {'health@QUERY': 0, 'app@QUERY': 1, 'app@LAUNCH': 2, 'email@LAUNCH': 3, 'riddle@QUERY': 4}
    with open(slot_vocab_file) as f:
        res = [i.lower().strip() for i in f.readlines() if len(i.strip()) != 0]
    slot_vocab = dict(zip(res, range(1, len(res)+1)))
    # slot_vocab示例 {'B-content': 0, 'B-endLoc_poi': 1, 'B-dynasty': 2, 'I-teleOperator': 3, 'I-headNum': 4}

    class Logger:
        def info(self, s):
            print(s)
    logger = Logger()
    tokz = AutoTokenizer.from_pretrained(bert_path)
    # tokz = BertTokenizer.from_pretrained(bert_path) 与上面的写法相同，后面可以跟路径也可以根据名字，如果是根据名字，它会自己写在一个，然后每次去取就好了
    # dataset = NLUDataset.make_dataset([data_file], tokz, cls_vocab, slot_vocab, logger)
    dataset = NLUDataset([data_file], tokz, cls_vocab, slot_vocab, logger) # 必须通过这个类调用，才有__getitem__

    pad = PadBatchSeq(tokz.pad_token_id) # 这里传入了一个pad_id, 就是补全而使用的pad_token_id
    # print(pad([dataset[i] for i in range(5)]))

