'''
Author: Aditya Jain
Date  : 21st May, 2021
About : concatenates labels and prediction for all the test points to plot confusion matrix
'''
import numpy as np

def confusion_matrix_data(prev_data, new_data):
	'''
	concatenates labels and prediction for all the test points to plot confusion matrix
	'''
	if prev_data == None:
		labels, preds    = new_data[0], new_data[1]
		return [np.array(labels), np.array(preds)]

	else:
		prev_data[0] = np.append(prev_data[0], new_data[0], axis=0)
		prev_data[1] = np.append(prev_data[1], new_data[1], axis=0)        
		return prev_data

    

## Plotting using seaborn
# sn.heatmap(cm_family, fmt="d", annot=True, xticklabels=family_list, yticklabels=family_list)
# plt.title('Family Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# experiment.log_figure(figure=plt)

# sn.heatmap(cm_genus, fmt="d", annot=True, xticklabels=genus_list, yticklabels=genus_list)
# plt.title('Genus Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# experiment.log_figure(figure=plt)

# sn.heatmap(cm_species, fmt="d", annot=True, xticklabels=species_list, yticklabels=species_list)
# plt.title('Species Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# experiment.log_figure(figure=plt)

# cm_species           = confusion_matrix(global_confusion_data_sp[0], global_confusion_data_sp[1], labels=range(len(species_list)))
# cm_genus             = confusion_matrix(global_confusion_data_g[0], global_confusion_data_g[1], labels=range(len(genus_list)))
# cm_family            = confusion_matrix(global_confusion_data_f[0], global_confusion_data_f[1], labels=range(len(family_list)))