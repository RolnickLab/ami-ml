# Assembling webdatasets

Please refer to the [documentation](https://docs.google.com/document/d/1JMbU7exXyaJicldYBgMszY6hgy1J22dCki-TOEgYE0o/edit?usp=sharing) for a detailed description of the pipeline for assembling training datasets using GBIF data.

## Overview

Here is an overview of the scripts available in this folder:

| script                 | function                                                                                  | multijob* | multiprocess** |
|------------------------|-------------------------------------------------------------------------------------------|-----------|----------------|
| `fetch_images.py`      | download images from URLs in observation metadata from a Darwin Core Archive (DwC-A) file | Yes       | Yes            |
| `verify_images.py`     | check images for errors                                                                   | No        | Yes            |
| `delete_images.py`     | delete a list of images                                                                   | No        | No             |
| `clean_dataset.py`     | filter out images to ensure the quality of the training data                              | No        | No             |
| `split_dataset.py`     | split dataset into training/validation/test sets                                          | No        | No             |
| `create_webdataset.py` | assemble final training set in webdataset format                                          | No        | No             |

*The script can be run using multiple Slurm job instances in parallel for faster execution.

**The script can use multiple processes for faster execution. Use the Slurm directive `--cpus-per-task` to define the number of processes per node and set the option `--num-workers` in the script.

<br/>

The purpose and use of each script is detailed in its docstring. Scripts' options can be listed
directly in the CLI with the `--help` flag:

```bash
python src/dataset_tools/fetch_images.py --help
```
