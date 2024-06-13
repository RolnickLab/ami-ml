## Description
Evaluation of the AMI-GBIF trained binary and fine-grained moth classifier on the AMI-Traps dataset.


## Overview
There are two main evaluation scripts: `binary_model_evaluation.py` and `fine_grained_model_evaluation.py`. The former evaluates the binary moth-nonmoth classifier, while the latter evaluates the fine-grained moth species classifier classifier. The remaining scripts are helper functions or generate one-time taxonomoy related files required for the evaluation.

Note:
- The `plots` directory has scripts to further analyze the evaluation results specific to the paper.
- The evaluation scripts can be run through the command line or submitted to a GPU cluster using a bash script.
