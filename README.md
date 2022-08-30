<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NLP Text Classification Model

Finetuning the [Graphcore Hugging Face Optimum](https://github.com/huggingface/optimum-graphcore) model to use on local datasets.

## requirements.txt

 **A list of all of a project's dependencies**. This includes the dependencies needed by the dependencies. It also contains the specific version of each dependency.

## run_glue.py

Main file with source code needed to run text classification model. Many attributes are provided in order to run this file and will be shown below.

The model is also able to use the library models for sequence classification on the GLUE benchmark:  [General Language Understanding Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the  [hub](https://huggingface.co/models)  and can also be used for a dataset hosted on our  [hub](https://huggingface.co/datasets)  or your own data in a csv or a JSON file (the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

    export TASK_NAME=sst2
    
    python run_glue.py \
      --model_name_or_path bert-base-cased \
      --ipu_config_name Graphcore/bert-base-ipu \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 3 \
      --output_dir ./output/$TASK_NAME/

where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

# Getting the dataset

For this example I have focused on using my own dataset to use on the Graphcore Hugging Face Model

To download the dataset example is very straightforward. 
Simply git clone the dataset from Hugging Face Datasets:

    git clone [https://huggingface.co/datasets/simana/textclassificationMNLI/](https://huggingface.co/datasets/simana/textclassificationMNLI/)

After performing this git command, you should be able to see a folder called TextClassificationMNLI containing the `test.csv , train.csv ` and ` validation.csv`



## Running the example 

Firstly you must make sure that you have installed Optimum.  You can do so using this command:

    pip install optimum[graphcore]

You must also ensure that you have downloaded the requirements.txt file to make sure that you are up to date with the necessary packages required to run the model:

    pip install -r requirements.txt

In order to run the model with the given example datasets, please follow the commands below:

    export TASK_NAME=mnli
    
    python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --ipu_config_name Graphcore/bert-base-ipu \
    --train_file textclassificationMNLI/train.csv \
    --test_file textclassificationMNLI/test.csv \
    --validation_file textclassificationMNLI/validation.csv \
    --do_train \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --pod_type pod8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./output/**classificationtest**/

After the model has successfully complied you should be able to see the output results within the `output/classificationtest` folder.
-->
