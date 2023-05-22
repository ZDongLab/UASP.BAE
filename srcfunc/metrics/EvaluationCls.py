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

class EvaluationCls():
    '''
    EvaluationC: for classification
    '''
    def __init__(self, names_labels_id: dict):
        self.names_labels_id = names_labels_id
        self.num_classes = len(names_labels_id.keys())
        self.id_labels_names={names_labels_id[name]: name for name in list(names_labels_id.keys())}
        #self.labels_name = list(self.labels_names.keys())
        #self.labels_id = list(self.labels_names.values())
        self.table = None
        self.cls_preds=[]
        self.cls_target=[]
        self.matrix = []


    def update_cls(self, preds, target, thresh=0.5):
        #print(preds, labels)
        self.cls_preds.append(preds)
        self.cls_target.append(target)
        #ConfusionMatrix
        ConMatrix = confusion_matrix(preds, target, self.num_classes, threshold=thresh, multilabel=False)
        self.matrix.append(ConMatrix.numpy())

    def metrics_table(self):
        cls_preds = torch.cat(self.cls_preds, 0)
        cls_target = torch.cat(self.cls_target, 0)
        AUCs=[]
        if self.num_classes==2:
            for id in list(self.id_labels_names.keys()):
                AUCs.append(auroc(cls_preds[:,id], cls_target, pos_label=id))
        else: 
            AUCs = auroc(cls_preds, cls_target, num_classes=self.num_classes)
        
        ClassNumber = [sum(cls_target==id) for id in list(self.id_labels_names.keys())]
        # precision, recall, specificity
        table = PrettyTable(['cls_name', "numbers","TP", "FP", "FN", "TN",
                             'precision', 'recall','Accuracy','f1-score',"AUC"])
        #table.field_names[' ', 'precision', 'recall','specificity','f1-score']
        matrix = sum(self.matrix)
        for id in list(self.id_labels_names.keys()):
            TP = matrix[id,id]
            FP = np.sum(matrix[id,:]) - TP
            FN = np.sum(matrix[:,id]) - TP
            TN = np.sum(matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 4) if TP + FP !=0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN !=0 else 0.
            Specificity = round(TN / (TN + FP), 4) if TN + FP !=0 else 0.
            Accuracy = round((TP+TN) / (TN + TP + FP + FN), 4) if TN + TP + FP + FN !=0 else 0.
            f1_score = round(2*TP/(2*TP + FN + FP), 4) if TP + FN + FP !=0 else 0.
            if self.num_classes==2:
                AUC = np.round(AUCs[id].numpy(),4)
            else:
                AUC = np.round(AUCs.numpy(),4)
            attri_num = ClassNumber[id].numpy()
            table.add_row([self.id_labels_names[id],attri_num, TP, FP, FN, TN, 
                           Precision, Recall, Accuracy, f1_score, AUC])
        self.table = table
        return table

    def plot_ROCAUC(self, ax, fpr, tpr, AUC, labelname):
        label = "{}:{}".format(str(labelname),round(AUC, 4))
        ax.plot(fpr, tpr, label=label)
        return ax

    def show_ROCAUC(self, plotname):
        cls_preds = torch.cat(self.cls_preds, 0)
        cls_target = torch.cat(self.cls_target, 0)
        fprs, tprs, thresholds = roc(cls_preds, cls_target, num_classes=self.num_classes, pos_label=1)
        AUCs = auroc(cls_preds, cls_target, num_classes=self.num_classes, pos_label=1, average="micro")

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