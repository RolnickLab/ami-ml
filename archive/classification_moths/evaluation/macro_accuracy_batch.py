'''
Author        : Aditya Jain
Date Started  : 26th May, 2021
About         : returns macro accuracy of a batch for different taxon levels for different top 'n' accuracies
'''

import torch
import json
import numpy as np
from operator import add

class MacroAccuracyBatch():
	def __init__(self, predictions, labels, label_info, taxon_hierar):
		'''
		Args:
			predictions  : model predictions for the batch
			labels       : ground truth for the batch 
			label_info   : contains conversion of numeric and string labels
			taxon_hierar : gives the genus and family of a species
		'''
		self.predictions  = predictions
		self.labels       = labels.cpu()
		self.taxon_hierar = taxon_hierar
		self.label_info   = label_info
		self.total        = len(self.labels)
		self.final_data   = {}
		
	
	def batch_accuracy(self):
		'''
		returns the final accuracy
		'''
		_, predict_indx_1  = torch.topk(self.predictions, 1)
		_, predict_indx_3  = torch.topk(self.predictions, 3)
		_, predict_indx_10 = torch.topk(self.predictions, 10)
		
		predict_indx_1     = np.array(predict_indx_1.cpu())
		predict_indx_3     = np.array(predict_indx_3.cpu())
		predict_indx_10    = np.array(predict_indx_10.cpu())
		
		##### species #######  
		species_dict       = {}  # key (label index) = [top1_c, top3_c, top10_c, total]
		
		for i in range(self.total):
			label_indx = int(self.labels[i])
			if label_indx in predict_indx_1[i]:
				if label_indx not in species_dict.keys():
					species_dict[label_indx] = [1, 1, 1, 1]
				else:
					species_dict[label_indx][0] += 1
					species_dict[label_indx][1] += 1
					species_dict[label_indx][2] += 1
					species_dict[label_indx][3] += 1
            
			elif label_indx in predict_indx_3[i]:
				if label_indx not in species_dict.keys():
					species_dict[label_indx] = [0, 1, 1, 1]
				else:
					species_dict[label_indx][1] += 1
					species_dict[label_indx][2] += 1
					species_dict[label_indx][3] += 1
            
			elif label_indx in predict_indx_10[i]:
				if label_indx not in species_dict.keys():
					species_dict[label_indx] = [0, 0, 1, 1]
				else:
					species_dict[label_indx][2] += 1
					species_dict[label_indx][3] += 1
			else:
				if label_indx not in species_dict.keys():
					species_dict[label_indx] = [0, 0, 0, 1]
				else:
					species_dict[label_indx][3] += 1
		
		self.final_data['species']  = species_dict
		#####################
		
		##### genus #######
		genus_labels          = self.genus_to_label_gt(self.species_to_genus_gt(self.label_to_spname_gt()))
		genus_predict_indx_1  = self.genus_to_label_pr(self.species_to_genus_pr(self.label_to_spname_pr(predict_indx_1)))
		genus_predict_indx_3  = self.genus_to_label_pr(self.species_to_genus_pr(self.label_to_spname_pr(predict_indx_3)))
		genus_predict_indx_10 = self.genus_to_label_pr(self.species_to_genus_pr(self.label_to_spname_pr(predict_indx_10)))
		
		genus_dict       = {}  # key (label index) = [top1_c, top3_c, top10_c, total]
		
		for i in range(self.total):
			label_indx = int(genus_labels[i])
			if label_indx in genus_predict_indx_1[i]:
				if label_indx not in genus_dict.keys():
					genus_dict[label_indx] = [1, 1, 1, 1]
				else:
					genus_dict[label_indx][0] += 1
					genus_dict[label_indx][1] += 1
					genus_dict[label_indx][2] += 1
					genus_dict[label_indx][3] += 1
            
			elif label_indx in genus_predict_indx_3[i]:
				if label_indx not in genus_dict.keys():
					genus_dict[label_indx] = [0, 1, 1, 1]
				else:
					genus_dict[label_indx][1] += 1
					genus_dict[label_indx][2] += 1
					genus_dict[label_indx][3] += 1
            
			elif label_indx in genus_predict_indx_10[i]:
				if label_indx not in genus_dict.keys():
					genus_dict[label_indx] = [0, 0, 1, 1]
				else:
					genus_dict[label_indx][2] += 1
					genus_dict[label_indx][3] += 1
					
			else:
				if label_indx not in genus_dict.keys():
					genus_dict[label_indx] = [0, 0, 0, 1]
				else:
					genus_dict[label_indx][3] += 1
		
		
		self.final_data['genus']  = genus_dict
		###################
		
		##### family #######
		family_labels          = self.family_to_label_gt(self.species_to_family_gt(self.label_to_spname_gt()))
		family_predict_indx_1  = self.family_to_label_pr(self.species_to_family_pr(self.label_to_spname_pr(predict_indx_1)))
		family_predict_indx_3  = self.family_to_label_pr(self.species_to_family_pr(self.label_to_spname_pr(predict_indx_3)))
		family_predict_indx_10 = self.family_to_label_pr(self.species_to_family_pr(self.label_to_spname_pr(predict_indx_10)))
		
		family_dict       = {}  # key (label index) = [top1_c, top3_c, top10_c, total]
		
		for i in range(self.total):
			label_indx = int(family_labels[i])
			if label_indx in family_predict_indx_1[i]:
				if label_indx not in family_dict.keys():
					family_dict[label_indx] = [1, 1, 1, 1]
				else:
					family_dict[label_indx][0] += 1
					family_dict[label_indx][1] += 1
					family_dict[label_indx][2] += 1
					family_dict[label_indx][3] += 1
            
			elif label_indx in family_predict_indx_3[i]:
				if label_indx not in family_dict.keys():
					family_dict[label_indx] = [0, 1, 1, 1]
				else:
					family_dict[label_indx][1] += 1
					family_dict[label_indx][2] += 1
					family_dict[label_indx][3] += 1
            
			elif label_indx in family_predict_indx_10[i]:
				if label_indx not in family_dict.keys():
					family_dict[label_indx] = [0, 0, 1, 1]
				else:
					family_dict[label_indx][2] += 1
					family_dict[label_indx][3] += 1
					
			else:
				if label_indx not in family_dict.keys():
					family_dict[label_indx] = [0, 0, 0, 1]
				else:
					family_dict[label_indx][3] += 1
		
		
		self.final_data['family']  = family_dict
		
		return self.final_data
		
	def label_to_spname_gt(self):
		'''
		returns species names from its numeric labels for the ground truth
		'''
		f            = open(self.label_info)
		label_info   = json.load(f)
		species_list = label_info['species_list']
		species_name = []    
		
		for index in self.labels:
			species_name.append(species_list[index])
			
		return species_name

	def species_to_genus_gt(self, species):
		'''
		returns the genus of a species for the ground truth
		'''
		f            = open(self.taxon_hierar)
		taxon_info   = json.load(f)
		genus_list   = []
    
		for item in species:
			genus_list.append(taxon_info[item][0])
        
		return genus_list
	
	def species_to_family_gt(self, species):
		'''
		returns the family of a species for the ground truth
		'''
		f            = open(self.taxon_hierar)
		taxon_info   = json.load(f)
		family_list   = []
    
		for item in species:
			family_list.append(taxon_info[item][1])
        
		return family_list
	
	def genus_to_label_gt(self, genus):
		'''
		returns numeric labels for a list of genus names for the ground truth
		'''
		f            = open(self.label_info)
		label_info   = json.load(f)
		genus_list   = label_info['genus_list']
		genus_label  = []
    
		for item in genus:
			genus_label.append(genus_list.index(item))
        
		return genus_label
	
	def family_to_label_gt(self, family):
		'''
		returns numeric labels for a list of family names for the ground truth
		'''
		f            = open(self.label_info)
		label_info   = json.load(f)
		family_list   = label_info['family_list']
		family_label  = []
    
		for item in family:
			family_label.append(family_list.index(item))
        
		return family_label
	
	def label_to_spname_pr(self, pred):
		'''
		returns species names from its numeric labels for the model prediction
		'''
		f            = open(self.label_info)
		label_info   = json.load(f)
		species_list = label_info['species_list']
		pred_species_name = [] 
        
		for batch in pred:
			species_name = []
			for index in batch:
				species_name.append(species_list[index])            
			pred_species_name.append(species_name)
            
		return pred_species_name 
	
	def species_to_genus_pr(self, species):
		'''
		returns the genus of a species for the model prediction
		'''
		f                 = open(self.taxon_hierar)
		taxon_info        = json.load(f)
		pred_genus_list   = []
    
		for batch in species:
			genus_list = []
			for item in batch:
				genus_list.append(taxon_info[item][0])
				
			pred_genus_list.append(genus_list)
        
		return pred_genus_list
	
	def species_to_family_pr(self, species):
		'''
		returns the family of a species for the model prediction
		'''
		f                 = open(self.taxon_hierar)
		taxon_info        = json.load(f)
		pred_family_list  = []
    
		for batch in species:
			family_list = []
			for item in batch:
				family_list.append(taxon_info[item][1])
				
			pred_family_list.append(family_list)
        
		return pred_family_list
	
	def genus_to_label_pr(self, genus):
		'''
		returns numeric labels for a list of genus names for the model prediction
		'''
		f                 = open(self.label_info)
		label_info        = json.load(f)
		genus_list        = label_info['genus_list']
		pred_genus_label  = []
    
		for batch in genus:
			genus_label = []
			for item in batch:
				genus_label.append(genus_list.index(item))
				
			pred_genus_label.append(genus_label)
        
		return pred_genus_label
	
	def family_to_label_pr(self, family):
		'''
		returns numeric labels for a list of family names for the model prediction
		'''
		f                 = open(self.label_info)
		label_info        = json.load(f)
		family_list       = label_info['family_list']
		pred_family_label = []
    
		for batch in family:
			family_label = []
			for item in batch:
				family_label.append(family_list.index(item))
				
			pred_family_label.append(family_label)
        
		return pred_family_label
	

def add_batch_macroacc(prev_data, new_data):
	'''
	adds the correct batch predictions for a global total
	'''
	if prev_data == None:
		return new_data
	else:
		for taxon in new_data.keys():
			tax_data = new_data[taxon]
			for key in tax_data.keys():
				if key not in prev_data[taxon].keys():
					prev_data[taxon][key] = new_data[taxon][key]
				else:
					prev_data[taxon][key] = list(map(add, prev_data[taxon][key], new_data[taxon][key]))
                      
	return prev_data
    
def final_macroacc(data):
	'''
	returns final macro accuracy for entire test set
	'''
	final_acc = {}
    
	for taxon in data.keys():
		tax_data = data[taxon]
		final_acc[taxon] = {}
		for key in tax_data.keys():
			final_acc[taxon][key] = [round(data[taxon][key][0]/data[taxon][key][3]*100,2), 
                                     round(data[taxon][key][1]/data[taxon][key][3]*100,2),
                                     round(data[taxon][key][2]/data[taxon][key][3]*100,2),
                                     data[taxon][key][3]]
	final_result = {}
    
	for taxon in final_acc.keys():        
		tax_data  = final_acc[taxon]   
		top1_tot  = 0
		top3_tot  = 0
		top10_tot = 0
		for key in tax_data.keys():
			top1_tot  += final_acc[taxon][key][0]
			top3_tot  += final_acc[taxon][key][1]
			top10_tot += final_acc[taxon][key][2]
            
		final_result["macro_"+taxon+"_top1"]  = round(top1_tot/len(tax_data),2)
		final_result["macro_"+taxon+"_top3"]  = round(top3_tot/len(tax_data),2)
		final_result["macro_"+taxon+"_top10"] = round(top10_tot/len(tax_data),2)

	return final_result, final_acc

    
def taxon_accuracy(data, label_info):
    '''
    returns top1 accuracy for every taxon at different ranks
    '''
    final_data          = {}
    final_data['About'] = '[Top1 Accuracy, Total Test Points] for each taxon at different ranks'
    
    ## family
    family_list         = label_info['family_list']
    f_data              = data['family']
    family_data         = {}
                             
    for key in f_data.keys():
        family_data[family_list[key]] = [f_data[key][0], f_data[key][3]]
    
    family_data          = dict(sorted(family_data.items(), key=lambda item: item[1], reverse=True))
    final_data['family'] = family_data
    
    ## genus
    genus_list           = label_info['genus_list']
    g_data               = data['genus']
    genus_data           = {}
                             
    for key in g_data.keys():
        genus_data[genus_list[key]] = [g_data[key][0], g_data[key][3]]
    
    genus_data           = dict(sorted(genus_data.items(), key=lambda item: item[1], reverse=True))
    final_data['genus']  = genus_data
    
    ## species
    species_list         = label_info['species_list']
    s_data               = data['species']
    species_data         = {}
                             
    for key in s_data.keys():
        species_data[species_list[key]] = [s_data[key][0], s_data[key][3]]
    
    species_data         = dict(sorted(species_data.items(), key=lambda item: item[1], reverse=True))
    final_data['species']= species_data
    
    return final_data    
    