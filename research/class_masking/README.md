## Class Masking
This repository contains code for masking a pre-trained model to only classify a subset of classes. 
For example, if you have a model that classifies 1000 classes, but you only want to classify 10 of them, you can use this code to mask the output layer to only classify those 10 classes.

As a use case, the Panama model is masked to only classify the Barro Colorado Island (BCI) species.

### Species of Interest
The python notebook `modifications_for_bci_classifier.ipynb` contains code to process a checklist and save the species of interest to a pickle file. Upon reading, the pickle should be a list of species names and the model will restrict predictions to only these species. An example is below:
```
species_of_interest = [
    'Artace cribrarius',
    'Dolichosomastis leucogrammica',
    'Hilarographa plectanodes',
    'Trosia dimas',
    'Asturodes fimbriauralis',
    'Megalopyge opercularis',
    'Perigramma vicina',
    ...
]
```

### Model Inference Class
The [`ModelInference`](https://github.com/RolnickLab/ami-ml/blob/main/src/classification/model_inference.py) class has been modified to include a species masking list. In addition to the species list `masking_list.pkl`, two json mapping files are required:

```
taxon_key_to_id_map = {
    '1732901': 0,
    '1743793': 1,
    '1748056': 2,
    '1751492': 3,
    '1751512': 4,
    '1751572': 5,
    '1751590': 6,
    ...
}
```


```
taxon_key_to_name_map = {
    '1732901': 'Artace cribrarius',
    '1743793': 'Mictopsichia hubneriana',
    '1748056': 'Hilarographa plectanodes',
    '1751492': 'Trosia dimas',
    '1751512': 'Trosia fallax',
    '1751572': 'Megalopyge opercularis',
    '1751590': 'Megalopyge tharops',
    ...
}
```


An example usage of class pruning is in `masking.py`.