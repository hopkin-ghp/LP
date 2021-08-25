# encoding='utf-8'
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


import matplotlib.pyplot as plt

data_path = r'F:\project_pycharm\LP\checkpoint\result(neg=1,reverse,testneg).log'
true_label = []
pred_label = []
f = open(data_path, 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    line = line.strip().split(',')
    line = [float(i) for i in line]
    true_label.append(int(line[0]))
    pred_label.append(int(1) if line[1] >= 0.5 else int(0))

acc = accuracy_score(true_label, pred_label)
print("准确率为%f" % acc)
precision = precision_score(true_label, pred_label)
print("精确率为%f" % precision)
recall = recall_score(true_label, pred_label)
print("召回率为%f" % recall)
f1 = f1_score(true_label, pred_label)
print("F1值为%f" % f1)


fpr, tpr, threshold = roc_curve(true_label,pred_label)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))

plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(fpr, tpr, color='darkorange',
lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例')
plt.ylabel('真正例')
plt.title('链接预测ROC曲线')
plt.legend(loc="lower right")
plt.show()