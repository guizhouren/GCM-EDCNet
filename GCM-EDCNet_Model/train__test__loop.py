import torch
from test_gpus import try_gpu
import numpy as np
from prettytable import PrettyTable

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity",'F1']
        Precision_list,ReCall_list,Specificity_list,F1_list =[],[],[],[]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2*(Precision*Recall)/(Precision+Recall),3) if Precision+Recall!=0 else 0
            Precision_list.append(Precision),ReCall_list.append(Recall),Specificity_list.append(Specificity),F1_list.append(F1)

            table.add_row([self.labels[i], Precision, Recall, Specificity,F1])
        avg_row = np.round([np.mean(Precision_list),np.mean(ReCall_list),np.mean(Specificity_list),np.mean(F1_list)],3)
        table.add_row(['avg',avg_row[0],avg_row[1],avg_row[2],avg_row[3]])
        print(self.matrix)
        print(table)
        return str(acc)



def train_loop(models,loss_fn,optimizer,train_iter,val_iter,epochs,device=try_gpu(2),acc=None,cl_type=5):
    losses_list =[]
    models.to(device)
    print('开始训练')
    max_acc = 0
    current_acc=0
    best_epoch =0
    for epoch in range(1,epochs+1):
        loss_train = 0.0
        for i,data in enumerate(train_iter,1):
            imgs,labels = data
            imgs,labels = imgs.to(device),labels.to(device)
            outputs = models(imgs)
            loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            loss_train+=loss.item()

            if i%100==0:
                print(f'Epoch:{epoch},i:{i},训练损失:{loss_train/100:.6f}')
                losses_list.append(round(loss_train/100,6))
                loss_train = 0.0

        current_acc=test_loop(models,val_iter,device)
        if acc !=None:
            if current_acc>=max_acc:
                max_acc = current_acc
                torch.save(models.state_dict(),'model_bestparams_'+str(cl_type)+'.pkl')
                best_epoch=epoch
    print('最佳轮数',best_epoch)
    return losses_list



def test_loop(models,val_iter,device=try_gpu(2),need_confusion=False,num_classes=5):
    correct = 0
    total = 0
    models.to(device)
    models.eval()
    if need_confusion:
        confusion = ConfusionMatrix(num_classes=num_classes,labels=[i for i in range(num_classes)])

    with torch.no_grad():
        for imgs,labels in val_iter:
            imgs,labels = imgs.to(device),labels.to(device)
            outputs = models(imgs)
            _,preds = torch.max(outputs,dim=1)
            correct += int((preds==labels).sum())
            total += labels.shape[0]
            if need_confusion:
                confusion.update(preds.cpu().numpy(), labels.cpu().numpy())
        if need_confusion:
            confusion.summary()

    print(f'精度:{correct/total*100:.3f}%')
    return np.round(correct/total*100,3)