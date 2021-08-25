import os

base_path = './'
data_path = os.path.join(base_path, 'data')


def attribute_analyse():
    path = os.path.join(data_path, 'attribute')
    len2cnt_dic = {}
    word2cnt_dic = {}
    attribute_set = set()
    for i in range(1, 10756):
        lines = open(os.path.join(path, str(i) + '.txt'), 'r', encoding='utf-8').readlines()[1:]
        lines = [line.strip() for line in lines]

        for word in lines:
            attribute_set.add(word)
            if word not in word2cnt_dic:
                word2cnt_dic[word] = 0
            word2cnt_dic[word] += 1

        length = len(lines)
        if length not in len2cnt_dic:
            len2cnt_dic[length] = 0
        len2cnt_dic[length] += 1

    cnt = [0 for i in range(10)]
    for key in word2cnt_dic.keys():
        value = word2cnt_dic[key]
        if value == 1:
            cnt[0] += 1
        elif value == 2:
            cnt[1] += 1
        elif value == 3:
            cnt[2] += 1
        elif value == 4:
            cnt[3] += 1
        elif value == 5:
            cnt[4] += 1
        elif 6 < value < 10:
            cnt[5] += 1
        elif 11 < value < 100:
            cnt[6] += 1
        elif 101 < value < 200:
            cnt[7] += 1
        elif 201 < value < 500:
            cnt[8] += 1
        elif 501 < value:
            cnt[9] += 1

    keys = []
    values = []
    for key in len2cnt_dic.keys():
        keys.append(key)
        values.append(len2cnt_dic[key])
    # print(word2cnt_dic)
    # print(len2cnt_dic)
    print(cnt)
    cnt = [i / len(word2cnt_dic) for i in cnt]
    print(cnt)
    print(len(word2cnt_dic))


if __name__ == '__main__':
    attribute_analyse()
