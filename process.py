import os

base_path = './'
data_path = os.path.join(base_path, 'data')

def read_entity():
    path = os.path.join(data_path, 'attribute')
    f = open(os.path.join(data_path, 'entity'), 'w', encoding='utf-8')
    for i in range(1, 10756):
        lines = open(os.path.join(path, str(i) + '.txt'), 'r', encoding='utf-8').readlines()[1:]
        lines = [line.strip() for line in lines]
        if len(lines) == 0:
            f.writelines(str(i) + '\t' + '填充字符' + '\n')
        else:
            f.writelines(str(i) + '\t' + '\t'.join(lines) + '\n')


def statistics():
    '''
    将频率出现次数为1的属性去掉
    :return:
    '''
    entity_path = os.path.join(data_path, 'entity')
    f_read = open(entity_path, 'r', encoding='utf-8')
    f_write = open(os.path.join(data_path, 'entityStatistics'), 'w', encoding='utf-8')
    lines = f_read.readlines()
    lines = [line.strip().split('\t') for line in lines]
    word2freq = {}
    for line in lines:
        for word in line[1:]:
            if word in word2freq:
                word2freq[word] += 1
            else:
                word2freq[word] = 1
    print(word2freq)

    # 统计
    # minCnt = 0
    # for key in word2freq.keys():
    #     if word2freq[key] == 1:
    #         minCnt += 1
    # print(len(word2freq), minCnt)
    for line in lines:
        tmp = []
        for word in line[1:]:
            if word2freq[word] != 1:
                tmp.append(word)
        f_write.writelines(line[0] + '\t' + '\t'.join(tmp) + '\n')


if __name__ == '__main__':
    read_entity()
    statistics()