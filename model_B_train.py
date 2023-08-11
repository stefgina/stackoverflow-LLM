# !pip install scikit-learn
# !pip install seaborn
# !pip install matplotlib
# !pip install numpy
# !pip install pandas
# !pip install transformers
# !pip install torch
# !pip install gdown
# !gdown --id 1Udrd9a944rJH0GxDhR6052gGNksb7rXO # df_eda.pkl from google drive



########################################### TRAIN PART ###########################################
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import seaborn as sns
import shutil, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import torch
import os


############################### CONFIG
MAX_LEN = 225
TRAIN_BATCH_SIZE = 36
VALID_BATCH_SIZE = 36
EPOCHS = 5
LEARNING_RATE = 1e-05

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }


################################ INIT
warnings.simplefilter("ignore")
sns.set_style("darkgrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


############################### CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


############################### JOIN TITLE + BODY
df = pd.read_pickle("df_eda.pkl")
df['Combo'] = df['Title'] + ". " + df['Body']


############################### BINARIZATION
mlb = MultiLabelBinarizer()
tag_df = pd.DataFrame(mlb.fit_transform(df['Tags']), columns=mlb.classes_, index=df.index)
class_names = mlb.classes_

############################### DATAFRAME HOUSEKEEPING
df = df.join(tag_df)
df = df.drop(columns='Tags')
df['target_list'] = df.iloc[:, 3:103].values.tolist()
df = df.drop(df.columns[3:103], axis=1)
df = df.drop(df.columns[0:2], axis=1)

# DEBUG
# print(df.head(2))
# print(df.shape)


############################### SPLIT
# cross checking that my train and test split is exatcly the same with
# the train test split i did for the model A.
# reason for the check is the difference in data structures (df vs array)
# (for example pd.sample(random_state=0) returns different split than sklearn for the same state)

# so the split is 80/20 for train-val/test
# and another 80/20 for train/val
# so train: 72%, val 8%, and test 20%

# Splitting the dataframe
train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=0)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=0)

train_dataset = train_dataset.reset_index(drop=True)
val_dataset = val_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)

# Xy_train, Xy_test = train_test_split(df, test_size = 0.2, random_state = 0)
# Xy_train, Xy_val = train_test_split(Xy_train, test_size = 0.1, random_state = 0)

# # Resetting the indices
# train_dataset = test_dataset.reset_index(drop=True)
# val_dataset = valid_dataset.reset_index(drop=True)


# DEBUG
# print(Xy_train.head(1))
# print(Xy_test.head(1))
# quit()

print("[PROGRAM]: full-set shape: {}".format(df.shape))
print("[PROGRAM]: train-set shape: {}".format(train_dataset.shape))
print("[PROGRAM]: val-set shape: {}".format(val_dataset.shape))
print("[PROGRAM]: test-set shape: {}".format(test_dataset.shape))


############################### TORCH DATASET
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.combo = dataframe['Combo']
        self.targets = self.data.target_list
        self.max_len = max_len

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, index):
        combo = str(self.combo[index])
        combo = " ".join(combo.split())

        inputs = self.tokenizer.encode_plus(
            combo,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


train_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)
test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

# DEBUG
# print(train_set[0])


############################### TORCH DATALOADER
training_loader = DataLoader(train_set, **train_params)
validation_loader = DataLoader(val_set, **test_params)
test_loader = DataLoader(test_set, **test_params)

len(training_loader)


############################### TRAIN FUNCS
# chckpoint and save funcs from here
# https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def load_ckp(checkpoint_fpath, model, optimizer):
    # load check point
    # initialize state_dict from checkpoint to model
    # initialize optimizer from checkpoint to optimizer
    # initialize valid_loss_min from checkpoint to valid_loss_min
    # return model, optimizer, epoch value, min validation loss

    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    # save checkpoint data to the path given, checkpoint_path
    # if it is a best model, min validation loss
    # copy that checkpoint file to best path given, best_model_path

    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)


############################### MODEL
# base : bert
# extra dropout + linear layer
# ending in 100 neurons, just like our classes
# after i extract the propabillities of each of the 100 neurons
# i select the proba >0.5 and bin the results to (0,1) (like sigmoid but manual)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 100)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)
print("[INFO]: model loaded to device")

############################### LOOP
# globals
val_targets=[]
val_outputs=[]
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

print("[INFO]: starting training")
def train_model(start_epochs,  n_epochs, valid_loss_min_input,
                training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, n_epochs+1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('[PROGRAM]: epoch', epoch)
        print('[PROGRAM]: TRAINING START')
        for batch_idx, data in enumerate(training_loader):
            print('[PROGRAM]: TRAINING batch ', batch_idx, ' /1420')
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        model.eval()
        print('[PROGRAM]: epoch', epoch)
        print('[PROGRAM]: VALIDATION START')

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                print('[PROGRAM]: VALIDATION batch ', batch_idx)
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            print('[PROGRAM]: VALIDATION END')
            # calculate average losses
            train_loss = train_loss/len(training_loader)
            valid_loss = valid_loss/len(validation_loader)
            print('[PROGRAM]: Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
                ))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            if valid_loss <= valid_loss_min:
                print('[PROGRAM]: Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

        print('[PROGRAM]: Epoch {}  Done \n'.format(epoch))


    return model


############################### DIRS
# check if the directory already exists
# if it does not exist, create the directory

directory_checkpoints = "/checkpoints"
directory_best_models = "/best_models"

if not os.path.exists(directory_checkpoints):
    os.makedirs(directory_checkpoints)
    print("Directory '/checkpoints' created successfully.")
else:
    print("Directory '/checkpoints' already exists.")

if not os.path.exists(directory_best_models):
    os.makedirs(directory_best_models)
    print("Directory '/best_models' created successfully.")
else:
    print("Directory '/best_models' already exists.")

checkpoint_path = "/checkpoints/current_checkpoint.pt"
best_model = "/best_models/best_model.pt"


############################### RUN
trained_model = train_model(1, EPOCHS, np.Inf, training_loader, validation_loader, model,
                      optimizer,checkpoint_path,best_model)


