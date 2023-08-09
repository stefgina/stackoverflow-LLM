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
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os


################################ INIT
warnings.simplefilter("ignore")
sns.set_style("darkgrid")


################################ DIRS
HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(HOME_DIR, "output")
print("[INFO]: setted up dirs")


################################ READS
df = pd.read_pickle(f"{DATA_DIR}/df_eda.pkl")

############################### JOIN TITLE + BODY
df['Combo'] = df['Title'] + ". " + df['Body']
print('[INFO]: title and body merged')

############################### INPUT VECTORIZATION
vectorizer = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)

X_tfidf = vectorizer.fit_transform(df['Combo'])
print('[INFO]: input vectorized')
print('[DEBUG]: merged vectorized shape: ', X_tfidf.shape)


############################### TARGET BINARIZATION
y = df['Tags']
multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(y)
y_classes_names = multilabel_binarizer.classes_
print('[DEBUG]: y_bin shape: ', y_bin.shape)
print('[DEBUG]: y_classes_names shape: ', y_classes_names.shape)
print('[INFO]: target binarized')


############################### SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

# DEBUG
# invtrain = vectorizer.inverse_transform(X_train)
# invtest = vectorizer.inverse_transform(X_test)
# print(invtrain[0])
# print(invtest[0])

############################### LABEL DISTRIBUTIONS (BEFORE/ AFTER SPLIT)
class_proportions_before_split = y_bin.mean(axis=0)
class_proportions_train = y_train.mean(axis=0)
class_proportions_test = y_test.mean(axis=0)

# DEBUG
# print("[PROGRAM]: label distribution DATASET: ", class_proportions_before_split)
# print("[PROGRAM]: label distribution TRAIN: ", class_proportions_train)
# print("[PROGRAM]: label distribution TEST: ", class_proportions_test)

len_labels = range(len(class_proportions_before_split))

# histogram for class proportions
plt.figure(figsize=(10, 6))
plt.bar(len_labels, class_proportions_before_split, label="Before Split")
plt.bar(len_labels, class_proportions_train, label="After Split (Train)", alpha=0.7)
plt.bar(len_labels, class_proportions_test, label="After Split (Test)", alpha=0.5)

plt.xlabel("Class Index")
plt.ylabel("Proportion")
plt.title("Class Proportions Before and After Train-Test Split")
plt.legend()
plt.tight_layout()
plt.show()


# curves for class proportions
plt.figure(figsize=(10, 6))
plt.plot(len_labels, class_proportions_before_split, label="Before Split")
plt.plot(len_labels, class_proportions_train, label="After Split (Train)", alpha=0.7)
plt.plot(len_labels, class_proportions_test, label="After Split (Test)", alpha=0.5)

plt.xlabel("Class Index")
plt.ylabel("Proportion")
plt.title("Class Proportions Before and After Train-Test Split")
plt.legend()
plt.tight_layout()
plt.show()


################################ METRICS
def score_avg(y_pred, y_test):
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    hamming = hamming_loss(y_test, y_pred)
    jacard = jaccard_score(y_test, y_pred, average='micro')

    print("[PROGRAM]: classifier -> LinearSVC")
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
                         columns=y_classes_names)


################################ MULTI-LABEL & MULTI-CLASS CLASSIFIER
# https://scikit-learn.org/stable/modules/multiclass.html
# based on the official website of sklearn, for multilabel & multiclass
# should use the MultiOutputClassifier

# ex. prediction (we predict more than 1 labels)
# [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# SVC with linear kernel is NOT the same as LinearSVC.
# SVC : 1/2||w||^2 + C SUM xi_i
# LinearSVC: 1/2||[w b]||^2 + C SUM xi_i
# ++ SVC returns propabillities (predict_proba), LinearSVC not. But SVC is 50x slower
# more here: https://stackoverflow.com/questions/33843981/under-what-parameters-are-svc-and-linearsvc-in-scikit-learn-equivalent/33844092#33844092

# https://stackoverflow.com/questions/64490621/linearsvc-and-roc-auc-score-for-a-multi-class-problem
# solving the problem of no propabillities in LinearSVC

classifier = CalibratedClassifierCV(LinearSVC(verbose=0))
clf = MultiOutputClassifier(classifier)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

# DEBUG
# print(type(y_pred))
# print(type(y_test))
# print(type(y_score))
# print(np.shape(y_pred))
# print(np.shape(y_test))
# print(np.shape(y_score))


################################# METRICS
metrics_avg = score_avg(y_pred, y_test)
metrics_per_tag = score_per_tag(y_pred, y_test)
print(metrics_per_tag)


################################ METRICS ON TOP TEN TAGS
top_ten_tags = ["javascript", "java", "c#", "php", "android", "jquery", "python", "html", "c++", "ios"]
print(metrics_per_tag[top_ten_tags])
print(metrics_per_tag[top_ten_tags].apply(np.mean, axis=1))


################################ MICRO-AVG PRECISION/RECALL
# https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
# based on comments on stackexchange
# Micro averaged 𝐵(∑𝑞𝑗=1𝑇𝑃𝑗,∑𝑞𝑗=1𝐹𝑃𝑗,∑𝑞𝑗=1𝑇𝑁𝑗,∑𝑞𝑗=1𝐹𝑁𝑗)

def plot_micro_average_roc(y_test, y_score):
    # setting up the plot
    # first aggregate all false positive rates
    # then interpolate all ROC curves at these points
    # finally average it and compute AUC
    # compute micro-average ROC curve and ROC area
    # plot ROC curve

    plt.figure(figsize=(10, 8))
    lw = 2 
    y_test_ravel = y_test.ravel()
    y_score_ravel = np.array([prob[:, 1] for prob in y_score]).T.ravel()
    fpr, tpr, _ = roc_curve(y_test_ravel, y_score_ravel)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='deeppink', linestyle='-', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
plot_micro_average_roc(y_test, y_score)


################################ MACRO-AVG PRECISION/RECALL
# https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
# based on comments on stackexchange
# macro averaged 1𝑞∑𝑞𝑗=1𝐵(𝑇𝑃𝑗,𝐹𝑃𝑗,𝑇𝑁𝑗,𝐹𝑁𝑗)

def plot_macro_average_roc(y_test, y_score, n_classes):
    # setting up the plot
    # first aggregate all false positive rates
    # then interpolate all ROC curves at these points
    # finally average it and compute AUC
    # compute macro-average ROC curve and ROC area
    # plot ROC curve

    plt.figure(figsize=(10, 8))
    lw = 2 
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[i][:, 1])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= n_classes
    roc_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr,
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='navy', linestyle='-', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")

    plt.show()


n_classes = y_test.shape[1]
plot_macro_average_roc(y_test, y_score, n_classes)


################################ FREQUENT TAGS PRECISION/RECALL

def plot_roc_curve(y_test, y_score, classes):
    plt.figure(figsize=(10, 8))
    lw = 2 
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[i][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 
                 lw=lw, 
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(label, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for TOP Tags')
    plt.legend(loc="lower right")
    plt.show()

top_tags_index = [list(y_classes_names).index(tag) for tag in top_ten_tags]
y_test_top_tags = y_test[:, top_tags_index]
y_score_top_tags = [y_score[i] for i in top_tags_index]
plot_roc_curve(y_test_top_tags, y_score_top_tags, top_ten_tags)


################################ ALL TAGS PRECISION/RECALL
def plot_roc_curve(y_test, y_score, classes):

    plt.figure(figsize=(10, 8))
    lw = 2  
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[i][:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 
                 lw=lw, 
                 # label='ROC curve of class {0} (area = {1:0.2f})'.format(label, roc_auc)
                 )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for ALL Tags')
    plt.show()

tags_index = [list(y_classes_names).index(tag) for tag in y_classes_names]
y_test_tags = y_test[:, tags_index]
y_score_tags = [y_score[i] for i in tags_index]
plot_roc_curve(y_test_tags, y_score_tags, y_classes_names)

