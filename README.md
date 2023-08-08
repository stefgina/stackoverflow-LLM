# Tag Prediction Stack Overflow (NLP)
```python
Author -> Stefanos Ginargyros
```

![stackoverflow](images/logo-stackoverflow.png)

## Introduction

In 2019 Stack Overflow realeased a [public dataset](https://www.kaggle.com/datasets/stackoverflow/stacksample) in Kaggle, named 10% of Stack Overflow Q&A. Its a very quality dataset, including text from 10% of Stack Overflow questions and answers on programming topics. In this project we are automatically predicting the tags of the questions, utilizing a variety of `NLP` tools and models. This is one of the demanding classification problems since it incorporates at the same time multi-label (cardinality >2) and multi-class (>1) targets. Fortunatelly enough, we have fresh tools 🤗 and ideas/ papers attacking these kind of problems, given there is enough compute! 

## Dataset

The raw dataset by default is split in 3 CSV files:

- `Questions.csv`: There is usefull information including cumulative question scores, ids, creation and closing datetimes and more importantly the **Title** and **Body** for each question.

- `Answers.csv`: Includes ids, and answers for each question. Every question can be pointed by many answer ids. Its many to one.

- `Tags.csv`: Includes ids, and tags for each question. Every question can be pointed by many tag ids. Its many to one as well.


## Installation

Okay so there are three ways of handling the installation process. 

- You can pip install everything and run `Locally` 

    ```bash
        # install deps
        pip install -r requirements.txt
    ```

- You can use the dockerfile of this repo and run everything through a `Container`. The commands for that would be:

    ```bash
        # build the image
        docker build -t docker-eda-mlmodel -f Dockerfile .
    ```

- Or you can skip completely any installation, and directy use my `Collab` notebooks and run the experiments. This includes training, finetuning and predicting utilizing a Large Language Model (BERT, distilBERT etc)

    [bert_fine_stackoverflow_v6_train.ipynb](https://colab.research.google.com/drive/1IlNwHCM2rWZqZMNiByoGzAq-M7p2FWui?usp=sharing)

    [bert_fine_stackoverflow_v6_infer.ipynb](https://colab.research.google.com/drive/18JgKJEwGVjYK3QisDOOe1PkHKEA5Btyy?usp=sharing)


## Usage Locally
Since you have installed all dependencies locally, this section will guide you through the experiments. Usually I choose to create conda virtual envs, when playing with new projects. So in this case:

```bash 
    # create the env, and activate
    conda create -n stackoverflow python=3.10
    conda activate stackoverflow
```

The basic usage locally is:

```bash
    # run eda on stack overflow data
    python3 eda.py 
```

```bash
    # predict tags  on stack overflow data, with ML model [cpu]
    python3 model_A_train_infer.py 
```

You will have to manually download the StackOverflow Dataset, if you go this way (locally). For your convience I have them uploaded and you can download them automatically with the following code. They will be extracted in the `-O` flag directory of the gdown command. For example I export them in the data folder (it must be an existing folder), since this is the folder all of my scripts are pointing.

```bash
    # fetch files from google drive
    gdown --id 1Udrd9a944rJH0GxDhR6052gGNksb7rXO -O data/df_eda.pkl
    gdown --id 1u8PWLs_SqSq0SMBXZSIB1LG59oror_B7 -O data/Questions.csv
    gdown --id 1ooskIp7eb7QOMeK1yJxXE1KkZoDARdfW -O data/Tags.csv
```
## Usage Docker
Don't stress about the Data, or the Dataset here, I have automated scripts downloading every part of the data you will need. Just run bellow lines:


```bash
    # run eda on stack overflow data
    docker run docker-eda-mlmodel python3 eda.py 
```

```bash
    # run ML model [cpu] on stack overflow data
    docker run docker-eda-mlmodel python3 model_A_train_infer.py 
```

## Usage Collab 

Follow my links, log-in to Google, and then choose as your runtime a GPU. (if you are subscriber you can even pick an A100, if not just a Tesla T4 for a limited time).

[bert_fine_stackoverflow_v6_train.ipynb](https://colab.research.google.com/drive/1IlNwHCM2rWZqZMNiByoGzAq-M7p2FWui?usp=sharing)

[bert_fine_stackoverflow_v6_infer.ipynb](https://colab.research.google.com/drive/18JgKJEwGVjYK3QisDOOe1PkHKEA5Btyy?usp=sharing)


## Project Structure 
The structure of the project follows classic NLP pipelines:
    
 - Exploratory Data Analysis on the whole Stack Overflow Dataset [`eda.py`]
    - Nulls, Duplicates
    - Frequent Target Analysis
    - Plots
    - Dataframe Operations (joins, chops)
    - Target Distribution
    - Loseless Dataset Shrinking (as possible)
    - Text Length Outliers (huge texts, encodings)
    - Stip Html (beatiful soup)
    - Accented characters
    - Special characters 
    - Lemmatization (playing, player -> play)
    - Expansion of contractions (ain't -> is not)

 - ML Model for Tag Prediction [`model_A_train_infer.py`]
    - Input Vectorization (Tfid)
    - Target Binarization (MultiLabelBinarizer)
    - Train/Val/Test Splits 
    - Target Distribution Cross Check
    - Model Selection (cpu currently)
    - Metrics (hamming, jaccard, f1, precision, recall)
    - Plots (AUC, ROC etc.)

 - LLM Model for Tag Prediction [`collab links`]
    - Label Binarization
    - Input Tokenization (BERT)
    - Dataset/Dataloader
    - Model (LLM + Classifier)
    - Train/Eval Loops
    - Metrics (micro-avg & tag frequent)


# Project Results 
Some interesting findings can be obtained through the Exploratory Data Analysis step [`eda.py`] and the [`model_A_train_infer.py`] for the ML[cpu] model.

- Through the EDA, the part that stand out to me are the frequencies of the tags plotted. 
    
    <img src="images/freq-tags.png" alt="drawing" width="480"/>

- Here are the distribution of plots `Before` (full dataset) and `After` the train/test split. In a) bars, and b) curves:


    <img src="images/bar_distr.png" alt="drawing" width="480"/>
    <img src="images/plot_distr.png" alt="drawing" width="480"/>

- Micro Average and Macro Average ROC Curves, and their AUC bottom right.

    <img src="images/micro-avg.png" alt="drawing" width="480"/>
    <img src="images/macro-avg.png" alt="drawing" width="480"/>

- ROC for the top 10 tags (from the first plot)

    <img src="images/roc_top.png" alt="drawing" width="480"/>

- ROC for all the tags 

    <img src="images/roc_all.png" alt="drawing" width="480"/>



## Numerical Benchmarks Micro-Average:
Numerical Benchmarks on the 10% of Stack Overflow Q&A Dataset for all the tags micro-averaged. Both of the models where fed the exact same input data, and tested on the exact same test-set. The dataset got carefully splitted in both cases 72% train, 8% validation and 20% test (while cross cheked the label distributions before and after the split in both cases).

| MODEL         | Precision     | Recall          |F-1          | True-Count    |Hamming Loss    |Jaccard score             | ~Time                 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------             |-------------          |
| LinearSVC     | 0.80141       | 0.4161        | 0.5478        | 0.9292        | 0.0108        | 0.3772                    | < 1 min (M1 Pro)      |
| BERT          | 0.8367        | 0.4596        | 0.5933        | 0.0099        | 0.0039        | 0.4218                    | > 5 hours (Tesla T4)  |


## Numerical Benchmarks Frequent Tags AVG:
Numerical Benchmarks on the 10% of Stack Overflow Q&A Dataset for the top 10 tags. These tags in descending frequency order are: 
```
    [javascript,java, c#, php, android, jquery, python, html, c++, ios ]
```
| MODEL         | Precision     | Recall          |F-1          | True-Count    |Hamming Loss    |Jaccard score     | ~Time                  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------     |-------------           |
| LinearSVC     | 0.8002        | 0.5096        | 0.6194        | 1115.3        |  0.0411       | 0.4636            | < 1 min (M1 Pro)       |
| BERT          | 0.8223        | 0.6647        | 0.7197        | 1115.3        | 0.0308        | 0.5875            | > 5 hours (Tesla T4)   |

