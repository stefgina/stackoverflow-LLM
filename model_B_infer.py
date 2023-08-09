# !pip install scikit-learn
# !pip install seaborn
# !pip install matplotlib
# !pip install numpy
# !pip install pandas
# !pip install transformers
# !pip install torch
# !pip install gdown
# !gdown --id 1--p20cXTZvk57GvPTxPmoylPtImvh0Vf # best_model.pt from google drive
# !gdown --id 1Udrd9a944rJH0GxDhR6052gGNksb7rXO # df_eda.pkl from google drive


########################################### INFERENCE SCRIPT ###########################################
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
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


################################ READS
df = pd.read_pickle("df_eda.pkl")


############################### CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


############################### JOIN TITLE + BODY
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


############################### TRAIN FUNCS
# chckpoint and save funcs from here (joe)
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
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
model,_,_,_ = load_ckp("best_model.pt", model, optimizer)
model.to(device)
print("[INFO]: BERT finetuned model loaded from best checkpoint")
print("[INFO]: model loaded to device")


################################ METRICS
def score_avg(y_pred, y_test):
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    hamming = hamming_loss(y_test, y_pred)
    jacard = jaccard_score(y_test, y_pred, average='micro')

    print("[PROGRAM]: classifier -> BERT finetuned")
    print("[PROGRAM]: avg precision: {}".format(precision))
    print("[PROGRAM]: avg recall: {}".format(recall))
    print("[PROGRAM]: avg f1-score: {}".format(f1))
    print("[PROGRAM]: avg hamming loss: {}".format(hamming))
    print("[PROGRAM]: avg jacard score: {}".format(jacard))

    return [precision, recall, f1, hamming, jacard]

def score_per_tag(y_pred, y_test):
    hamming = []
    jaccard = []
    precision, recall, fscore, support = score(y_test, y_pred)
    for i, (test, pred) in enumerate(zip(y_test.T, y_pred.T)):
        hamming.append(hamming_loss(test, pred))
        jaccard.append(jaccard_score(test,pred))

    # DEBUG
    # print(len(precision))
    # print(len(recall))
    # print(len(fscore))
    # print(len(support))
    # print(len(hamming))
    # print(len(jaccard))
    # print(len(y_classes))

    return pd.DataFrame(data=[precision, recall, fscore, support, hamming, jaccard],
                         index=["Precision", "Recall", "F-1 score", "True count", "Hamming loss", "Jaccard score"],
                         columns=mlb.classes_)


################################ INFERENCE TEST-SET
model.eval()
y_test = []
y_pred = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids)
        y_test.extend(targets.cpu().detach().numpy().tolist())
        y_pred.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        print("[PROGRAM]: INFERENCE BATCH ", batch_idx," /446")

# applying hard map of probas into (0,1)
y_pred = (np.array(y_pred) > 0.5).astype(int)
y_pred = np.array(y_pred)
y_test = np.array(y_test)


################################# METRICS (micro-average)
print("[INFO]: computing micro-average metrics for all tags")
metrics_avg = score_avg(y_pred, y_test)
metrics_per_tag = score_per_tag(y_pred, y_test)


################################ METRICS ON TOP TEN TAGS
top_ten_tags = ["javascript", "java", "c#", "php", "android", "jquery", "python", "html", "c++", "ios"]
print("[INFO]: computing top-ten tag metrics")
print(metrics_per_tag[top_ten_tags])
print("[INFO]: computing top-ten tag metrics averaged")
print(metrics_per_tag[top_ten_tags].apply(np.mean, axis=0))
