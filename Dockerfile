# base image
FROM jupyter/scipy-notebook

# set the working directory in the container
WORKDIR /stack_overflow_tag_prediction

# make the dirs in the container
RUN mkdir data
RUN mkdir output

RUN pip install gdown

# fetch files from google drive
RUN gdown --id 1Udrd9a944rJH0GxDhR6052gGNksb7rXO -O data/df_eda.pkl
RUN gdown --id 1u8PWLs_SqSq0SMBXZSIB1LG59oror_B7 -O data/Questions.csv
RUN gdown --id 1ooskIp7eb7QOMeK1yJxXE1KkZoDARdfW -O data/Tags.csv

# place files in container
# COPY /best_models/best_model.pt ./best_models/best_model.pt
# COPY /data/df_eda.pkl ./data/df_eda.pkl
# COPY /data/Questions.csv ./data/Questions.csv
# COPY /data/Tags.csv ./data/Tags.csv

# place scripts in container
COPY eda.py ./eda.py
COPY model_A_train_infer.py ./model_A_train_infer.py
COPY model_B_train.py ./model_B_train.py
COPY model_B_infer.py ./model_B_infer.py

# place requirements in container
COPY requirements.txt ./requirements.txt

# place readme in container
COPY README.md ./README.md

# install deps in container
RUN pip install -r requirements.txt
