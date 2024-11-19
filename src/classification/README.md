# Model Training


A (mostly) generic script for training vision classification models. Mostly generic becuase the code currently accepts only webdataset files as input for training.

The script can be run from anywhere within the `ami-ml` folder using `ami-classification train-model --help`.

To-do things for next sprint:
- Calculate macro-averaged accuracy
- Upload species checklist with individual taxa accuracy
- Expose the trained model using an API
- Option to train with non-webdataset format datasets  