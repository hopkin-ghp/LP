import os
import torch
from utils import TrainDataset, BidirectionalOneShotIterator, negdata, ValidDataset, TestDataset
from utils import read_data, wordtoindex, entitytoindex, data_path, base_path, model_path
from torch.utils.data import DataLoader
from model import ConvE
import argparse

device = torch.device('cuda')


def arg():
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=512, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1000)')
    parser.add_argument('--neg_size', type=int, default=1, help='number of negative samples for one sample')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--embedding_dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding_shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden_drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input_drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat_drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr_decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader_threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--use_bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden_size', type=int, default=19968, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--use_inverse', type=bool, default=False, help='inverse the knowledge graph')

    args = parser.parse_args()

    return args


def ranking_and_hits(model, test_iterator, param, f):
    print()
    print('*' * 50)
    print(param)
    print()
    total = 0
    true_cnt = 0
    for i, data in enumerate(test_iterator):
        e1, e2, attr1, attr2, label = data
        # torch.Size([64]) torch.Size([64]) torch.Size([64, 1237]) torch.Size([64, 1334]) torch.Size([64])
        e1 = e1.long().to(device)
        e2 = e2.long().to(device)
        attr1 = attr1.long().to(device)
        attr2 = attr2.long().to(device)
        pred = model.forward(e1, e2, attr1, attr2)
        pred = pred.view((-1)).tolist()
        label = label.tolist()
        total += len(pred)
        for score in pred:
            if score >= 0.5:
                true_cnt += 1

    acc = true_cnt / total
    print("Accuracy: %s" % acc)
    print('*' * 50)
    f.writelines('*' * 50 + '\n')
    f.writelines(param + '\n')
    f.writelines("Accuracy: %s" % acc + '\n')
    f.writelines('*' * 50 + '\n')

    return acc


def ranking_and_hits1(model, test_iterator, param, f):
    cnt = 0
    for i, data in enumerate(test_iterator):
        e1, e2, attr1, attr2, label = data
        # torch.Size([64]) torch.Size([64]) torch.Size([64, 1237]) torch.Size([64, 1334]) torch.Size([64])
        e1 = e1.long().to(device)
        e2 = e2.long().to(device)
        attr1 = attr1.long().to(device)
        attr2 = attr2.long().to(device)
        pred = model.forward(e1, e2, attr1, attr2)
        pred = pred.view((-1)).tolist()
        label = label.tolist()
        for i, data in enumerate(pred):
            if cnt < 16854:
                true_label = 1
            else:
                true_label = 0
            pred_score = data
            f.writelines(str(true_label) + ',' + str(pred_score) + '\n')
            cnt += 1


def main(args):

    # read data
    train_data = read_data(os.path.join(data_path, 'train.csv'))
    dev_data = read_data(os.path.join(data_path, 'dev.csv'))
    test_data = read_data(os.path.join(data_path, 'test.csv'))
    test_data_neg = read_data(os.path.join(data_path, 'test_neg.csv'))

    # process data
    word2index = wordtoindex()
    entity2index = entitytoindex(word2index)

    # process negative samples
    data = [train_data, dev_data, test_data, word2index, entity2index]
    neg = negdata(data, args.neg_size, args.use_inverse)
    e1_list, e2_list, attr1_list, attr2_list, label_list = None, None, None, None, None
    if args.neg_size == 0:
        e1_list, e2_list, attr1_list, attr2_list, label_list = neg.no_negdata_function()
    else:
        e1_list, e2_list, attr1_list, attr2_list, label_list = neg.negdata_function()
    assert len(e1_list) == len(e2_list) == len(attr1_list) == len(attr2_list) == len(label_list)

    print('**************************************')
    print()
    print("information of datasets: ")
    print("the number of triplets in train: %s" % (len(train_data)))
    print("the number of triplets in valid: %s" % (len(dev_data)))
    print("the number of triplets in test: %s" % (len(test_data)))
    print("the number of entities: %s" % (len(entity2index)))
    print("the number of attributes: %s" % (len(word2index)))
    print()
    print('**************************************')

    train_iterator = DataLoader(
        TrainDataset(e1_list, e2_list, attr1_list, attr2_list, label_list),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    valid_iterator = DataLoader(
        ValidDataset([dev_data, word2index, entity2index]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ValidDataset.collate_fn
    )
    test_iterator = DataLoader(
        TestDataset([test_data + test_data_neg, word2index, entity2index, len(test_data)]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TestDataset.collate_fn
    )
    # test_iterator_neg = DataLoader(
    #     ValidDataset([test_data_neg, word2index, entity2index]),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=ValidDataset.collate_fn
    # )

    model = ConvE(args, len(entity2index)+1, len(word2index)+1)
    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )

    log_file = os.path.join(model_path, 'result(neg=1).log')
    f = open(log_file, 'w', encoding='utf-8')
    print("start training : ")


    if False:
        dev_best_score = 0
        best_model = 0
        best_score = 0
        for epoch in range(args.epochs):
            model.train()
            res = 0
            cnt = 0
            for i, data in enumerate(train_iterator):
                e1, e2, attr1, attr2, label = data
                # torch.Size([64]) torch.Size([64]) torch.Size([64, 1237]) torch.Size([64, 1334]) torch.Size([64])
                e1 = e1.long().to(device)
                e2 = e2.long().to(device)
                attr1 = attr1.long().to(device)
                attr2 = attr2.long().to(device)
                label = label.float().to(device)
                label = label.unsqueeze(1).to(device)

                optimizer.zero_grad()
                # label smoothing
                # e2_multi = ((1.0 - args.label_smoothing) * e2_multi) + (1.0 / e2_multi.size(1))

                pred = model.forward(e1, e2, attr1, attr2)
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                cnt += 1

                res += loss.item()
            print("Epoch %d | loss: %f" % (epoch, res / cnt))
            f.writelines("Epoch %d | loss: %f" % (epoch, res / cnt) + '\n')

            model.eval()
            with torch.no_grad():
                if epoch % 5 == 0 or epoch <= 0:
                    current_score = ranking_and_hits(model, valid_iterator, 'dev_evaluation', f)
                    if dev_best_score <= current_score:
                        best_model = epoch
                        dev_best_score = current_score
                        save_path = os.path.join(model_path, str(epoch) + '.checkpoint')
                        print('saving to {0}'.format(save_path))
                        torch.save(model.state_dict(), save_path)
                        best_score = ranking_and_hits(model, test_iterator, 'test_evaluation', f)



        print("finish training!")
        print("best epoch is %s, best score is %s" % (best_model, best_score))
        f.writelines('\n' + '\n')
        f.writelines("finish training!" + '\n')
        f.writelines("best epoch is %s, best score is %s" % (best_model, best_score))
        f.close()
    else:
        model.load_state_dict(torch.load("./checkpoint/35.checkpoint"))
        log_file = os.path.join(model_path, 'result(neg=1,reverse,testneg).log')
        f = open(log_file, 'w', encoding='utf-8')
        model.eval()
        ranking_and_hits1(model, test_iterator, 'test dataset with negtive samples', f)


if __name__ == '__main__':
    args = arg()
    main(args)
