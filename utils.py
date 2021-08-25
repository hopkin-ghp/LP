import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

base_path = './'
data_path = os.path.join(base_path, 'data')
model_path = os.path.join(base_path, 'checkpoint')
if not os.path.exists(model_path):
    os.makedirs(model_path)


class negdata():
    def __init__(self, data, neg_size, use_inverse):
        """
        Dataset for training, inherits `torch.utils.data.Dataset`.
        Args:
            data_reader: DataReader,
            neg_size: int, negative sample size.
        """
        triples = data[0]
        self.triples = []
        if use_inverse:
            for i in triples:
                self.triples.append(i)
                self.triples.append([i[1], i[0]])
        else:
            self.triples = triples
        self.dev_data = data[1]
        self.test_data = data[2]
        self.word2index = data[3]
        self.entity2index = data[4]
        self.len = len(self.triples)
        self.num_entity = len(self.entity2index)
        self.num_word = len(self.word2index)
        self.neg_size = neg_size

        self.h_map, self.t_map, self.h_freq, self.t_freq = self.two_tuple_count()

    def no_negdata_function(self):
        e1 = []
        e2 = []
        attr1 = []
        attr2 = []
        label = []
        for idx in range(len(self.triples)):
            pos_triple = self.triples[idx]
            e1.append(pos_triple[0])
            e2.append(pos_triple[1])
            attr1.append(self.entity2index[pos_triple[0]])
            attr2.append(self.entity2index[pos_triple[1]])
            label.append(1)
        return e1, e2, attr1, attr2, label

    def negdata_function(self):
        e1 = []
        e2 = []
        attr1 = []
        attr2 = []
        attr = []
        label = []
        for idx in range(len(self.triples)):
            triples = []
            pos_triple = self.triples[idx]
            head, tail = pos_triple

            neg_triples_head = []
            neg_size_head = 0
            while neg_size_head < self.neg_size:
                neg_triples_tmp = np.random.randint(self.num_entity+1, size=self.neg_size * 2)
                mask = np.in1d(
                    neg_triples_tmp,
                    self.t_map[tail],
                    assume_unique=True,
                    invert=True
                )
                neg_triples_tmp = neg_triples_tmp[mask]
                neg_triples_head.append(neg_triples_tmp)
                neg_size_head += neg_triples_tmp.size

            neg_triples_tail = []
            neg_size_tail = 0
            while neg_size_tail < self.neg_size:
                neg_triples_tmp = np.random.randint(self.num_entity, size=self.neg_size * 2)
                mask = np.in1d(
                    neg_triples_tmp,
                    self.h_map[head],
                    assume_unique=True,
                    invert=True
                )
                neg_triples_tmp = neg_triples_tmp[mask]
                neg_triples_tail.append(neg_triples_tmp)
                neg_size_tail += neg_triples_tmp.size

            neg_triples_head = np.concatenate(neg_triples_head)[:self.neg_size]
            neg_triples_tail = np.concatenate(neg_triples_tail)[:self.neg_size]

            # print(pos_triple, neg_triples_head, neg_triples_tail)

            triples.append(pos_triple)
            e1.append(pos_triple[0])
            e2.append(pos_triple[1])
            for ele in neg_triples_head:
                triples.append([ele, pos_triple[1]])
                e1.append(ele)
                e2.append(pos_triple[1])
            for ele in neg_triples_tail:
                triples.append([pos_triple[0], ele])
                e1.append(pos_triple[0])
                e2.append(ele)

            for x in range(2 * self.neg_size + 1):
                tmp = [self.entity2index[triples[x][0]], self.entity2index[triples[x][1]]]
                attr.append(tmp)
                attr1.append(self.entity2index[triples[x][0]])
                attr2.append(self.entity2index[triples[x][1]])
            label.append(1)
            for i in range(2 * self.neg_size):
                label.append(0)
        return e1, e2, attr1, attr2, label

    def two_tuple_count(self):
        """
        Return two dict:
        dict({h: [t1, t2, ...]}),
        dict({t: [h1, h2, ...]}),
        """
        h_map = {}
        h_freq = {}
        t_map = {}
        t_freq = {}

        init_cnt = 3
        for head, tail in self.triples:
            if head not in h_map.keys():
                h_map[head] = set()
                h_map[head].add(0)

            if tail not in t_map.keys():
                t_map[tail] = set()
                t_map[tail].add(0)

            if head not in h_freq.keys():
                h_freq[head] = init_cnt

            if tail not in t_freq.keys():
                t_freq[tail] = init_cnt

            h_map[head].add(tail)
            t_map[tail].add(head)
            h_freq[head] += 1
            t_freq[tail] += 1

        for key in t_map.keys():
            t_map[key] = np.array(list(t_map[key]))

        for key in h_map.keys():
            h_map[key] = np.array(list(h_map[key]))

        return h_map, t_map, h_freq, t_freq

class TrainDataset(Dataset):
    def __init__(self, e1, e2, attr1, attr2, label):

        self.e1 = e1
        self.len = len(self.e1)
        self.e2 = e2
        self.attr1 = attr1
        self.attr2 = attr2
        self.label = label

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        Returns a positive sample and `self.neg_size` negative samples.
        """

        return self.e1[idx], self.e2[idx], self.attr1[idx], self.attr2[idx], self.label[idx]

    @staticmethod
    def collate_fn(data):
        e1 = []
        e2 = []
        attr1 = []
        attr2 = []
        label = []
        attr1_len = float('-inf')
        attr2_len = float('-inf')
        for i in data:
            attr1_len = attr1_len if attr1_len > len(i[2]) else len(i[2])
            attr2_len = attr2_len if attr2_len > len(i[3]) else len(i[3])
        for i, elem in enumerate(data):
            e1.append(elem[0])
            e2.append(elem[1])
            attr1.append(np.pad(elem[2], (0, attr1_len - len(elem[2])), 'constant', constant_values=20646).tolist())
            attr2.append(np.pad(elem[3], (0, attr2_len - len(elem[3])), 'constant', constant_values=20646).tolist())
            label.append(elem[4])

        return torch.tensor(e1), torch.tensor(e2), torch.tensor(attr1), torch.tensor(attr2), torch.tensor(label)

class ValidDataset(Dataset):
    def __init__(self, data):
        self.triples = data[0]
        self.word2index = data[1]
        self.entity2index = data[2]
        self.len = len(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):

        e1, e2 = self.triples[idx]
        e1, e2 = int(e1), int(e2)
        attr1 = self.entity2index[e1]
        attr2 = self.entity2index[e2]

        return e1, e2, attr1, attr2

    @staticmethod
    def collate_fn(data):

        e1 = []
        e2 = []
        attr1 = []
        attr2 = []
        attr1_len = float('-inf')
        attr2_len = float('-inf')
        for i in data:
            attr1_len = attr1_len if attr1_len > len(i[2]) else len(i[2])
            attr2_len = attr2_len if attr2_len > len(i[3]) else len(i[3])
        for i, elem in enumerate(data):
            e1.append(elem[0])
            e2.append(elem[1])
            attr1.append(np.pad(elem[2], (0, attr1_len - len(elem[2])), 'constant', constant_values=20646).tolist())
            attr2.append(np.pad(elem[3], (0, attr2_len - len(elem[3])), 'constant', constant_values=20646).tolist())

        return torch.tensor(e1), torch.tensor(e2), torch.tensor(attr1), torch.tensor(attr2)


class TestDataset(Dataset):
    def __init__(self, data):
        self.triples = data[0]
        self.word2index = data[1]
        self.entity2index = data[2]
        self.bound = data[3]
        self.len = len(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):

        e1, e2 = self.triples[idx]
        e1, e2 = int(e1), int(e2)
        attr1 = self.entity2index[e1]
        attr2 = self.entity2index[e2]

        return e1, e2, attr1, attr2

    @staticmethod
    def collate_fn(data):

        e1 = []
        e2 = []
        attr1 = []
        attr2 = []
        label = []
        attr1_len = float('-inf')
        attr2_len = float('-inf')
        for i in data:
            attr1_len = attr1_len if attr1_len > len(i[2]) else len(i[2])
            attr2_len = attr2_len if attr2_len > len(i[3]) else len(i[3])
        for i, elem in enumerate(data):
            label.append(1) if i < 16854 else label.append(0)
            e1.append(elem[0])
            e2.append(elem[1])
            attr1.append(np.pad(elem[2], (0, attr1_len - len(elem[2])), 'constant', constant_values=20646).tolist())
            attr2.append(np.pad(elem[3], (0, attr2_len - len(elem[3])), 'constant', constant_values=20646).tolist())

        return torch.tensor(e1), torch.tensor(e2), torch.tensor(attr1), torch.tensor(attr2), torch.tensor(label)


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

def read_data(path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    lines = [line.strip().split(',') for line in lines]
    lines = [[int(i) for i in line] for line in lines]
    return lines

def wordtoindex():
    f = open(os.path.join(data_path, 'entityStatistics'), 'r', encoding='utf-8')
    lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    word2index = {}
    cnt = 0
    for line in lines:
        for word in line[1:]:
            if word not in word2index:
                word2index[word] = cnt
                cnt += 1
    return word2index

def entitytoindex(word2index):
    f = open(os.path.join(data_path, 'entityStatistics'), 'r', encoding='utf-8')
    f_write = open(os.path.join(data_path, 'entity2index'), 'w', encoding='utf-8')
    lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    entity2index = {}
    for line in lines:
        tmp = []
        for word in line[1:]:
            tmp.append(word2index[word])
        entity2index[int(line[0])] = tmp
        string = ''
        for i in tmp:
            string += str(i) + '\t'
        f_write.writelines(line[0] + '\t' + string + '\n')
    return entity2index

def generate_neg():

    '''
    生成测试集负样本
    :return:空
    '''

    # read data
    train_data = read_data(os.path.join(data_path, 'train.csv'))
    dev_data = read_data(os.path.join(data_path, 'dev.csv'))
    test_data = read_data(os.path.join(data_path, 'test.csv'))
    test_data_neg_file = os.path.join(data_path, 'data/test_neg.csv')
    f = open(test_data_neg_file, 'w', encoding='utf-8')
    total_data = []
    for triple in train_data + test_data + dev_data:
        temp = []
        temp.append(triple[1])
        temp.append(triple[0])
        total_data.append(triple)
        total_data.append(temp)
    entity_set = set()
    for triple in total_data:
        entity_set.add(triple[0])
        entity_set.add(triple[1])
    entity_set = list(entity_set)
    neg_number = len(test_data)
    neg_data = []
    for triple in test_data:
        temp = []
        while(True):
            a = random.randint(1, len(entity_set)-1)
            temp = [triple[0], entity_set[a]]
            if temp not in total_data:
                neg_data.append(temp)
                temp = [str(i) for i in temp]
                f.writelines(','.join(temp) + '\n')
                break
