import imp, os
from turtle import color
from cv2 import threshold
import numpy as np
from prettytable import PrettyTable
#from sklearn.metrics import multilabel_confusion_matrix
import torch
from torchmetrics.functional import auroc, roc, confusion_matrix
#from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
'''
multiclass
metric_collection = MetricCollection([
    Accuracy(),
    Precision(num_classes=3, average='macro'),
    Recall(num_classes=3, average='macro')
])
print(metric_collection(preds, target))
f1 = F1Score(num_classes=3)
f1(preds, target)

confmat = ConfusionMatrix(num_classes=2)
confmat(preds, target)


roc = ROC(num_classes=4)
fpr, tpr, thresholds = roc(pred, target)
# binary case
auroc = AUROC(pos_label=1)
auroc(preds, target)
#multiclass case
auroc = AUROC(num_classes=3)
auroc(preds, target)

#========multilabel
confmat = ConfusionMatrix(num_classes=3, multilabel=True)
confmat(preds, target)

roc = ROC(num_classes=3, pos_label=1)
fpr, tpr, thresholds = roc(pred, target)
'''

class EvaluationMul():
    '''
    EvaluationC: for classification
    '''
    def __init__(self, labels_list: list):
        self.num_classes = len(labels_list)
        self.labels_dict={id : name for id, name in enumerate(labels_list)}
        self.labels_name = list(self.labels_dict.values())
        self.labels_id = list(self.labels_dict.keys())
        self.table = None
        self.multi_preds=[]
        self.multi_target=[]
        self.matrix = []


    def update_mul(self, preds, target, thresh=0.5):
        #print(preds, labels)
        self.multi_preds.append(preds)
        self.multi_target.append(target)
        #ConfusionMatrix
        ConMatrix = confusion_matrix(preds, target, self.num_classes, threshold=thresh, multilabel=True)
        self.matrix.append(ConMatrix.numpy())

    def metrics_table(self):
        multi_preds = torch.cat(self.multi_preds, 0)
        multi_target = torch.cat(self.multi_target, 0)
        AUCs = []
        ClassNumber = []
        for id in self.labels_id:
            AUCs.append(auroc(multi_preds[:,id], multi_target[:,id], pos_label=1))
            ClassNumber.append(sum(multi_target[:,id]))
        # precision, recall, specificity
        table = PrettyTable(['attributes', "numbers","TP", "FP", "FN", "TN",
                             'precision', 'recall','specificity','Accuracy','f1-score',"AUC"])
        #table.field_names[' ', 'precision', 'recall','specificity','f1-score']
        matrix = sum(self.matrix)
        for id in self.labels_id:
            TP = matrix[id][1,1]
            FP = matrix[id][0,1]
            FN = matrix[id][1,0]
            TN = matrix[id][0,0]
            Precision = round(TP / (TP + FP), 4) if TP + FP !=0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN !=0 else 0.
            Specificity = round(TN / (TN + FP), 4) if TN + FP !=0 else 0.
            Accuracy = round((TP+TN) / (TN + TP + FP + FN), 4) if TN + TP + FP + FN !=0 else 0.
            f1_score = round(2*TP/(2*TP + FN + FP), 4) if TP + FN + FP !=0 else 0.
            AUC = np.round(AUCs[id].numpy(),4)
            attri_num = ClassNumber[id].numpy()
            table.add_row([self.labels_name[id],attri_num, TP, FP, FN, TN, 
                           Precision, Recall, Specificity, Accuracy, f1_score, AUC])
        self.table = table
        return table

    def plot_ROCAUC(self, ax, fpr, tpr, AUC, labelname):
        label = "{}:{}".format(str(labelname),round(AUC, 4))
        ax.plot(fpr, tpr, label=label)
        return ax

    def show_ROCAUC(self, plotname):
        multi_preds = torch.cat(self.multi_preds, 0)
        multi_target = torch.cat(self.multi_target, 0)
        fprs, tprs, thresholds = roc(multi_preds, multi_target, num_classes=self.num_classes, pos_label=1)
        AUCs = auroc(multi_preds, multi_target, num_classes=self.num_classes, pos_label=1, average="micro")

        fig, ax = plt.subplots()
        for fpr, tpr, AUC, labelname in zip(fprs, tprs, AUCs, self.labels_name):
            ax = self.plot_ROCAUC(ax, fpr.numpy(), tpr.numpy(), AUC.numpy(), labelname)
        ax.plot((0,1), (0,1), ":", color="grey")
        ax.set_xlim(-0.01,1.01)
        ax.set_ylim(-0.01,1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        plt.savefig(os.path.join('./outputs', plotname+'.png'), format='png',dpi=600)
        eps_fig = plt.gcf()
        eps_fig.savefig(os.path.join('./outputs', plotname + '.eps'), format='eps', dpi=60)
        plt.show()