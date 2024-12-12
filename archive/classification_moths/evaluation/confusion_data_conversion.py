'''
Author: Aditya Jain
Date  : 18th May, 2021
About : returns species label data converted to genus and family
'''

import torch
import json
import numpy as np

class ConfusionDataConvert():
	def __init__(self, predictions, labels, label_info, taxon_hierar):
		'''
		Args:
			predictions  : model predictions for the batch
			labels       : ground truth for the batch 
			label_info   : contains conversion of numeric and string labels
			taxon_hierar : gives the genus and family of a species
		'''
		self.predictions  = predictions
		self.labels       = labels
		self.taxon_hierar = taxon_hierar
		self.label_info   = label_info
		self.total        = len(self.labels)
		self.final_data   = {}
		
	
	def converted_data(self):
		'''
		returns the final accuracy
		'''
		_, predict_indx_1  = torch.topk(self.predictions, 1)
		
		# genus
		genus_labels          = self.genus_to_label_gt(self.species_to_genus_gt(self.label_to_spname_gt()))
		genus_predict_indx_1  = self.genus_to_label_pr(self.species_to_genus_pr(self.label_to_spname_pr(predict_indx_1)))
		
		# family
		family_labels          = self.family_to_label_gt(self.species_to_family_gt(self.label_to_spname_gt()))
		family_predict_indx_1  = self.family_to_label_pr(self.species_to_family_pr(self.label_to_spname_pr(predict_indx_1)))
		
		return np.array(self.labels.cpu()), np.array(predict_indx_1.cpu()), genus_labels, genus_predict_indx_1, family_labels, family_predict_indx_1 
		
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

    