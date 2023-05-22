#import imp, os
import importlib as imp
from turtle import color
from cv2 import threshold
import numpy as np
from prettytable import PrettyTable
#from sklearn.metrics import multilabel_confusion_matrix
import torch
from torchmetrics.functional import auroc, roc, confusion_matrix
from torchmetrics.functional import mean_absolute_error as MAE
from torchmetrics.functional import mean_squared_error as MSE
from torchmetrics.functional import mean_absolute_percentage_error as MAPE
#from torchmetrics.functional import mean_squared_error as MSE

#from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt

class EvaluationMAE():
    '''
    EvaluationC: for classification
    '''
    def __init__(self, names_labels_id: dict):
        # names_labels_id = {"male":0, "female":1}
        self.names_labels_id = names_labels_id
        self.num_classes = len(names_labels_id.keys())

        self.id_labels_names={names_labels_id[name]: name for name in list(names_labels_id.keys())}
        #self.labels_name = list(self.labels_names.keys())
        #self.labels_id = list(self.labels_names.values())
        self.table = None
        self.cls_preds=[]
        self.cls_target=[]
        #self.matrix = []


    def update_mae(self, preds, target, thresh=0.5):
        #print(preds, labels)
        self.cls_preds.append(preds)
        self.cls_target.append(target)
        #ConfusionMatrix
        #ConMatrix = confusion_matrix(preds, target, self.num_classes, threshold=thresh, multilabel=False)
        #self.matrix.append(ConMatrix.numpy())

    def metrics_table(self):
        cls_preds = torch.cat(self.cls_preds, 0)
        cls_preds_int = torch.cat(self.cls_preds, 0).int()
        cls_target = torch.cat(self.cls_target, 0).unsqueeze(1)
        cls_target_int = torch.cat(self.cls_target, 0).int().unsqueeze(1)

        ClassNumber = {id:sum(cls_target_int==id) for id in list(self.id_labels_names.keys())}
        # precision, recall, specificity
        table = PrettyTable(['ages', "attri_num",
                            "ctaMAE", "ctaMSE", "ctaRMSE", "ctaMAPE", 
                            "intMAE", "intMSE", "intRMSE", "intMAPE"])
        #table.field_names[' ', 'precision', 'recall','specificity','f1-score']
        #matrix = sum(self.matrix)
        for idx in list(self.id_labels_names.keys()):
            indexs = cls_target_int==idx
            ctaMAE = MAE(cls_preds.masked_select(indexs), cls_target.masked_select(indexs)).numpy()
            ctaMSE = MSE(cls_preds.masked_select(indexs), cls_target.masked_select(indexs)).numpy()
            ctaRMSE = np.sqrt(ctaMSE)
            ctaMAPE = MAPE(cls_preds.masked_select(indexs), cls_target.masked_select(indexs)).numpy()

            intMAE = MAE(cls_preds_int.masked_select(indexs), cls_target_int.masked_select(indexs)).numpy()
            intMSE = MSE(cls_preds_int.masked_select(indexs), cls_target_int.masked_select(indexs)).numpy()
            intRMSE = np.sqrt(intMSE)
            intMAPE = MAPE(cls_preds_int.masked_select(indexs), cls_target_int.masked_select(indexs)).numpy()
            
            attri_num = ClassNumber[idx].numpy()[0]
            table.add_row([idx, attri_num,  
                           np.round(ctaMAE,4), np.round(ctaMSE,4), np.round(ctaRMSE,4), np.round(ctaMAPE,4),
                           np.round(intMAE,4), np.round(intMSE,4), np.round(intRMSE,4), np.round(intMAPE,4)])


        actaMAE = MAE(cls_preds, cls_target).numpy()
        actaMSE = MSE(cls_preds, cls_target).numpy()
        actaRMSE = np.sqrt(actaMSE)
        actaMAPE = MAPE(cls_preds, cls_target).numpy()

        aintMAE = MAE(cls_preds_int, cls_target_int).numpy()
        aintMSE = MSE(cls_preds_int, cls_target_int).numpy()
        aintRMSE = np.sqrt(aintMSE)
        aintMAPE = MAPE(cls_preds_int, cls_target_int).numpy()

        attri_num = torch.cat(list(ClassNumber.values()), 0).sum().numpy()
        table.add_row(["all",attri_num,  
                        np.round(actaMAE,4), np.round(actaMSE,4), np.round(actaRMSE,4), np.round(actaMAPE,4),
                        np.round(aintMAE,4), np.round(aintMSE,4), np.round(aintRMSE,4), np.round(aintMAPE,4)])

        self.table = table
        return table, actaMAE

    def boxplots(self, savename="",abs=False):
        cls_preds = torch.cat(self.cls_preds, 0)
        cls_preds_int = torch.cat(self.cls_preds, 0).int()
        #cls_target = torch.cat(self.cls_target, 0).unsqueeze(1)
        cls_target_int = torch.cat(self.cls_target, 0).int().unsqueeze(1)

        ClassNumber = {id:sum(cls_target_int==id) for id in list(self.id_labels_names.keys())}

        #all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
        all_data = []
        all_data_int =[]
        for idx in list(self.id_labels_names.keys()):
            indexs = cls_target_int==idx
            all_data.append((cls_preds.masked_select(indexs).numpy()-idx).tolist())
            all_data_int.append((cls_preds_int.masked_select(indexs).numpy()-idx).tolist())

        labels = list(self.id_labels_names.keys())

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))

        # rectangular box plot
        bplot1 = ax1.boxplot(all_data,
                            notch=True,  # notch shape
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=labels)  # will be used to label x-ticks
        ax1.set_title('Continuous age prediction')

        # notch shape box plot
        bplot2 = ax2.boxplot(all_data_int,
                            notch=True,  # notch shape
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=labels)  # will be used to label x-ticks
        ax2.set_title('Stage age prediction')

        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen',
                  'pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen',
                  'pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
        for bplot in (bplot1, bplot2):
            for idx, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(colors[idx])

        # adding horizontal grid lines
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True)
            ax.set_xlabel('Age range')
            ax.set_ylabel('Observed values')

        if savename:
            #plt.show()
            plt.rcParams['savefig.dpi'] = 300
            plt.savefig(savename+"_boxplots.pdf")    


    def adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value


    def set_axis_style(self, ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')
        
    def violinplots(self,savename):
        cls_preds = torch.cat(self.cls_preds, 0)
        cls_preds_int = torch.cat(self.cls_preds, 0).int()
        #cls_target = torch.cat(self.cls_target, 0).unsqueeze(1)
        cls_target_int = torch.cat(self.cls_target, 0).int().unsqueeze(1)

        ClassNumber = {id:sum(cls_target_int==id) for id in list(self.id_labels_names.keys())}

        #all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
        all_data = []
        all_data_int =[]
        for idx in list(self.id_labels_names.keys()):
            indexs = cls_target_int==idx
            all_data.append((cls_preds.masked_select(indexs).numpy()-idx).tolist())
            all_data_int.append((cls_preds_int.masked_select(indexs).numpy()-idx).tolist())

        labels = list(self.names_labels_id.keys())

        # create test data
        #np.random.seed(19680801)
        #data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 8), sharey=True)

        ax1.set_title('Continuous age prediction')
        ax1.set_ylabel('Observed values')
        ax1.violinplot(all_data)

        ax2.set_title('Stage age prediction')
        parts = ax2.violinplot(
                all_data_int, showmeans=False, showmedians=False,
                showextrema=False)

        for pc in parts['bodies']: #'pink', 'lightblue', 'lightgreen'
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('lightgreen')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = [],[],[]
        for data_l in all_data_int:    
            q1, ms, q3 = np.percentile([data_l], [25, 50, 75], axis=1)
            quartile1.append(q1[0])
            medians.append(ms[0])
            quartile3.append(q3[0])
        

        whiskers = np.array([
            self.adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(all_data_int, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # set style for the axes
        #labels = ['A', 'B', 'C', 'D']
        for ax in [ax1, ax2]:
            self.set_axis_style(ax, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        if savename:
            #plt.show()
            plt.rcParams['savefig.dpi'] = 300
            plt.savefig(savename+"_violinplot.pdf")  
