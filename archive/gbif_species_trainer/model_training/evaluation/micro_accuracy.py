"""
Author         : Aditya Jain
Last modified  : May 23rd, 2023
About          : Calculates micro accuracy of a batch for different taxon levels for different top 'n' accuracies
"""

import torch
import json


class MicroAccuracyBatch:
    def __init__(self, predictions, labels, label_info, taxon_hierar):
        """
        Args:
                predictions  : model predictions for the batch
                labels       : ground truth for the batch
                label_info   : contains conversion of numeric and string labels
                taxon_hierar : gives the genus and family of a species
        """
        self.predictions = predictions
        self.labels = labels
        self.taxon_hierar = taxon_hierar
        self.label_info = label_info
        self.total = len(self.labels)
        self.final_data = {}

    def batch_accuracy(self):
        """
        returns the final accuracy
        """
        _, predict_indx_1 = torch.topk(self.predictions, 1)
        _, predict_indx_3 = torch.topk(self.predictions, 3)
        _, predict_indx_10 = torch.topk(self.predictions, 10)

        # Species
        correct_1 = 0
        correct_3 = 0
        correct_10 = 0
        for i in range(self.total):
            if self.labels[i] in predict_indx_1[i]:
                correct_1 += 1

            if self.labels[i] in predict_indx_3[i]:
                correct_3 += 1

            if self.labels[i] in predict_indx_10[i]:
                correct_10 += 1

        self.final_data["micro_species_top1"] = correct_1
        self.final_data["micro_species_top3"] = correct_3
        self.final_data["micro_species_top10"] = correct_10

        # Genus
        genus_labels = self.genus_to_label_gt(
            self.species_to_genus_gt(self.label_to_species_name_gt())
        )
        genus_predict_indx_1 = self.genus_to_label_pr(
            self.species_to_genus_pr(self.label_to_species_name_pr(predict_indx_1))
        )
        genus_predict_indx_3 = self.genus_to_label_pr(
            self.species_to_genus_pr(self.label_to_species_name_pr(predict_indx_3))
        )
        genus_predict_indx_10 = self.genus_to_label_pr(
            self.species_to_genus_pr(self.label_to_species_name_pr(predict_indx_10))
        )

        correct_1 = 0
        correct_3 = 0
        correct_10 = 0
        for i in range(self.total):
            if genus_labels[i] in genus_predict_indx_1[i]:
                correct_1 += 1

            if genus_labels[i] in genus_predict_indx_3[i]:
                correct_3 += 1

            if genus_labels[i] in genus_predict_indx_10[i]:
                correct_10 += 1

        self.final_data["micro_genus_top1"] = correct_1
        self.final_data["micro_genus_top3"] = correct_3
        self.final_data["micro_genus_top10"] = correct_10

        # Family
        family_labels = self.family_to_label_gt(
            self.species_to_family_gt(self.label_to_species_name_gt())
        )
        family_predict_indx_1 = self.family_to_label_pr(
            self.species_to_family_pr(self.label_to_species_name_pr(predict_indx_1))
        )
        family_predict_indx_3 = self.family_to_label_pr(
            self.species_to_family_pr(self.label_to_species_name_pr(predict_indx_3))
        )
        family_predict_indx_10 = self.family_to_label_pr(
            self.species_to_family_pr(self.label_to_species_name_pr(predict_indx_10))
        )

        correct_1 = 0
        correct_3 = 0
        correct_10 = 0
        for i in range(self.total):
            if family_labels[i] in family_predict_indx_1[i]:
                correct_1 += 1

            if family_labels[i] in family_predict_indx_3[i]:
                correct_3 += 1

            if family_labels[i] in family_predict_indx_10[i]:
                correct_10 += 1

        self.final_data["micro_family_top1"] = correct_1
        self.final_data["micro_family_top3"] = correct_3
        self.final_data["micro_family_top10"] = correct_10

        self.final_data["total_points"] = self.total

        return self.final_data

    def label_to_species_name_gt(self):
        """
        returns species names from its numeric labels for the ground truth
        """
        f = open(self.label_info)
        label_info = json.load(f)
        species_list = label_info["species"]
        species_name = []

        for index in self.labels:
            species_name.append(species_list[index])

        return species_name

    def species_to_genus_gt(self, species):
        """
        returns the genus of a species for the ground truth
        """
        f = open(self.taxon_hierar)
        taxon_info = json.load(f)
        genus_list = []

        for item in species:
            genus_list.append(taxon_info[item][0])

        return genus_list

    def species_to_family_gt(self, species):
        """
        returns the family of a species for the ground truth
        """
        f = open(self.taxon_hierar)
        taxon_info = json.load(f)
        family_list = []

        for item in species:
            family_list.append(taxon_info[item][1])

        return family_list

    def genus_to_label_gt(self, genus):
        """
        returns numeric labels for a list of genus names for the ground truth
        """
        f = open(self.label_info)
        label_info = json.load(f)
        genus_list = label_info["genus"]
        genus_label = []

        for item in genus:
            genus_label.append(genus_list.index(item))

        return torch.LongTensor(genus_label)

    def family_to_label_gt(self, family):
        """
        returns numeric labels for a list of family names for the ground truth
        """
        f = open(self.label_info)
        label_info = json.load(f)
        family_list = label_info["family"]
        family_label = []

        for item in family:
            family_label.append(family_list.index(item))

        return torch.LongTensor(family_label)

    def label_to_species_name_pr(self, pred):
        """
        returns species names from its numeric labels for the model prediction
        """
        f = open(self.label_info)
        label_info = json.load(f)
        species_list = label_info["species"]
        pred_species_name = []

        for batch in pred:
            species_name = []
            for index in batch:
                species_name.append(species_list[index])
            pred_species_name.append(species_name)

        return pred_species_name

    def species_to_genus_pr(self, species):
        """
        returns the genus of a species for the model prediction
        """
        f = open(self.taxon_hierar)
        taxon_info = json.load(f)
        pred_genus_list = []

        for batch in species:
            genus_list = []
            for item in batch:
                genus_list.append(taxon_info[item][0])

            pred_genus_list.append(genus_list)

        return pred_genus_list

    def species_to_family_pr(self, species):
        """
        returns the family of a species for the model prediction
        """
        f = open(self.taxon_hierar)
        taxon_info = json.load(f)
        pred_family_list = []

        for batch in species:
            family_list = []
            for item in batch:
                family_list.append(taxon_info[item][1])

            pred_family_list.append(family_list)

        return pred_family_list

    def genus_to_label_pr(self, genus):
        """
        returns numeric labels for a list of genus names for the model prediction
        """
        f = open(self.label_info)
        label_info = json.load(f)
        genus_list = label_info["genus"]
        pred_genus_label = []

        for batch in genus:
            genus_label = []
            for item in batch:
                genus_label.append(genus_list.index(item))

            pred_genus_label.append(genus_label)

        return torch.LongTensor(pred_genus_label)

    def family_to_label_pr(self, family):
        """
        returns numeric labels for a list of family names for the model prediction
        """
        f = open(self.label_info)
        label_info = json.load(f)
        family_list = label_info["family"]
        pred_family_label = []

        for batch in family:
            family_label = []
            for item in batch:
                family_label.append(family_list.index(item))

            pred_family_label.append(family_label)

        return torch.LongTensor(pred_family_label)


def add_batch_microacc(prev_data, new_data):
    """
    adds the correct batch predictions for a global total
    """
    if prev_data == None:
        return new_data
    else:
        for key in new_data.keys():
            prev_data[key] += new_data[key]
        return prev_data


def final_micro_accuracy(data):
    """
    returns final micro accuracy for entire test set
    """
    final_acc = {}

    for key in data.keys():
        if key != "total_points":
            final_acc[key] = round(data[key] / data["total_points"] * 100, 2)

    return final_acc
