import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_words):
        super(ConvE, self).__init__()
        self.emb_entity = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_word = torch.nn.Embedding(num_words, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(args.embedding_dim)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        self.register_parameter('weight', Parameter(torch.zeros((args.embedding_dim, 1))))

    def init(self):
        xavier_normal_(self.emb_entity.weight.data)
        xavier_normal_(self.emb_word.weight.data)

    def forward(self, e1, e2, attr1, attr2):
        e1_embedded = self.emb_entity(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e2_embedded = self.emb_entity(e2).view(-1, 1, self.emb_dim1, self.emb_dim2)

        attr1_embedded = self.emb_word(attr1)
        attr1_embedded = torch.sum(attr1_embedded, 1) / attr1_embedded.size()[1]
        attr1_embedded = attr1_embedded.view(-1, 1, self.emb_dim1, self.emb_dim2)
        attr2_embedded = self.emb_word(attr2)
        attr2_embedded = torch.sum(attr2_embedded, 1) / attr2_embedded.size()[1]
        attr2_embedded = attr2_embedded.view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, attr1_embedded, e2_embedded, attr2_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x += self.b.expand_as(x)
        x = torch.mm(x, self.weight)
        # x = torch.mm(x, self.emb_entity.weight.transpose(1, 0))
        pred = torch.sigmoid(x)

        return pred

# input = torch.rand((1, 1, 200, 200))
#
# con1 = torch.nn.Conv2d(1, 1, 5, 2, 1)
# input_con1 = con1(input)
# print(input_con1.size())
#
# pool1 = torch.nn.MaxPool2d(3, 1, 0)
# input_pool1 = pool1(input_con1)
# print(input_pool1.size())
#
#
# pool2 = torch.nn.MaxPool2d(3, 1, 1)
# input_pool2 = pool2(input_pool1)
# print(input_pool2.size())
