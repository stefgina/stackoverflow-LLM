
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
import pandas as pd
import unicodedata
import nltk
import re
import os


################################ INIT
# python -m spacy download en_core_web_sm
warnings.simplefilter("ignore")
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
# nlp = spacy.load('en_core_web_sm')
stopword_list = stopwords.words('english')
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
lemma=WordNetLemmatizer()
selected_tags = None

################################ INIT
warnings.simplefilter("ignore")
pd.set_option('display.max_rows', None)


################################ DIRS
HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(HOME_DIR, "output")
print("[INFO]: setted up dirs")


################################ READS
questions_df = pd.read_csv(os.path.join(DATA_DIR, "Questions.csv"), encoding="ISO-8859-1", parse_dates=["CreationDate", "ClosedDate"])
print(f"[PROGRAM]: [questions] number of rows: {questions_df.shape[0]}")
print(f"[PROGRAM]: [questions] number of columns: {questions_df.shape[1]}")
print(f"[INFO]: [questions] loaded in memory")

tags_df = pd.read_csv(os.path.join(DATA_DIR, "Tags.csv"), encoding="ISO-8859-1")
print(f"[PROGRAM]: [tags] number of rows: {tags_df.shape[0]}")
print(f"[PROGRAM]: [tags] number of columns: {tags_df.shape[1]}")
print(f"[INFO]: [tags] loaded in memory")


################################ SIZE CHECK (debug)
print("[DEBUG] questions df size: ", questions_df.shape)
print("[DEBUG] tags df size: ", tags_df.shape)


################################ DROP DUPLICATES
questions_df.drop_duplicates()
tags_df.drop_duplicates()
print("[INFO]: dropping duplicates")


################################ DROP NULL
questions_df.dropna()
tags_df.dropna()
print("[INFO]: dropping nulls")


################################ DOMINANT-FREQUENT TAGS PLOTTING
# tags can be multiple for each question
# lets count them and find the most common ones

n_dominant_tags = 20
tag_value_counts = tags_df["Tag"].value_counts()
dominant_tags = tag_value_counts.head(n_dominant_tags)
dominant_tags_barplot = sns.barplot(x=dominant_tags.index, y=dominant_tags.values)
plt.xticks(rotation=90)
plt.show()

tags_string = ' '.join(dominant_tags.index.values)
print("[PROGRAM]: ex. 10 dominant tags: ", tags_string)


################################ SIZE CHECK (debug)
print("[DEBUG] questions df size: ", len(questions_df))
print("[DEBUG] tags df size: ", len(tags_df))


################################ PRE-JOIN TREATMENT 
# drop unused columns and merge tags per question
# group rows per question id

questions_df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)

tags_df['Tag'] = tags_df['Tag'].astype(str)
grouped_tags = tags_df.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags})
grouped_tags_final.head()
grouped_tags.reset_index()
print("[INFO]: preparation for join [questions] and [tags]")


################################ JOIN
# reset indexes
# merge

questions_df = questions_df.reset_index(drop=True)
grouped_tags_final = grouped_tags_final.reset_index(drop=True)
df = questions_df.merge(grouped_tags_final, on='Id')
print("[INFO]: join done [questions] and [tags] into [df]")


################################# LOW SCORE ENTRIES CHOP (<5)
good_scores = df['Score'] >= 5
df = df[good_scores]

print("[INFO]: low score entries chopped")


################################ SIZE CHECK
print("[DEBUG]: merged df size: ", df.shape)


################################ DOMINANT TAGS v2
n_of_classes = 100
df['Tags'] = df['Tags'].apply(lambda x: x.split())
flat_list = [item for sublist in df['Tags'].values for item in sublist]
keywords = nltk.FreqDist(flat_list)
keywords = nltk.FreqDist(keywords)
frequencies_words = keywords.most_common(n_of_classes)
selected_tags = [word[0] for word in frequencies_words]

def most_common(tags):
    tags_filtered = []
    for i in range(0, len(tags)):
        if tags[i] in selected_tags:
            tags_filtered.append(tags[i])
    return tags_filtered


################################ NON IMPORTANT TAGS CHOP
df['Tags'] = df['Tags'].apply(lambda x: most_common(x))
print("[INFO]: non important tag entries chopped")


################################ SIZE CHECK
print("[DEBUG]: chopped df size: ", df.shape)


################################ EMPTY TAGS CHOP
df['Tags'] = df['Tags'].apply(lambda x: x if len(x)>0 else None)
df = df.dropna()
print("[INFO]: empty tags chopped")


################################ SIZE CHECK
print("[DEBUG]: chopped df size: ", df.shape)


################################ tag counts
df["TagCount"] = df["Tags"].apply(len)


################################ STATS 
print(f'[PROGRAM]: each question has min {df["TagCount"].min()} tags')
print(f'[PROGRAM]: each question has max {df["TagCount"].max()} tags')
print(f'[PROGRAM]: each question has avg {df["TagCount"].mean()} tags')


################################ SHRINK DATAFRAME (throw ids, scores etc)
df.drop(columns=['Id', 'Score', 'TagCount'], inplace=True)


################################ SIZE CHECK
print("[DEBUG]: shrinked df size: ", df.shape)


################################ EXAMPLE OF FINAL DF
print(df.head(5))


# ################################ EXPORT
# df.to_pickle(f"{PROCESSED_DATA_DIR}/df_eda1.pkl")
# print("[INFO]: exported eda.pkl to disc")


################################ STANDARDIZE COLUMN NAMES
# df.columns = df.columns.str.lower()


################################ TEXT LENGTH OUTLIERS
# min title length is 9 and is fine
# max title length is 189 and has encoding errors (TODO: maybe i can filter out these later)
# min body length is 18 and its fine
# max body length is 46489 and full of html, we have to clean.

min_title_length, max_title_length = df["Title"].str.len().min(), df["Title"].str.len().max()
min_body_length, max_body_length = df["Body"].str.len().min(), df["Body"].str.len().max()
print(f"[PROGRAM]: min_title_length: {min_title_length}, max_title_length: {max_title_length}")
print(f"[PROGRAM]: min_body_length: {min_body_length}, max_body_length: {max_body_length}")

min_title_outlier = df[df["Title"].str.len() == min_title_length]
max_title_outlier = df[df["Title"].str.len() == max_title_length]

print(min_title_outlier)
print(max_title_outlier)

min_body_outlier = df[df["Body"].str.len() == min_body_length]
max_body_outlier = df[df["Body"].str.len() == max_body_length]

print(min_body_outlier)
print(max_body_outlier)

print(f"[INFO]: text outliers computed")


################################ CLEANING FUNCTIONS
def strip_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9#\s]'
    text = re.sub(pattern, '', text)
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None
  
def lemmatize_text(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def expand_contractions(text):
    text = text.lower()
    text = re.sub(r"ain't", "is not ", text)
    text = re.sub(r"aren't", "are not ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"can't've", "cannot have ", text)
    text = re.sub(r"'cause", "because ", text)
    text = re.sub(r"could've", "could have ", text)
    text = re.sub(r"couldn't", "could not ", text)
    text = re.sub(r"couldn't've", "could not have ", text)
    text = re.sub(r"didn't", "did not ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"hadn't", "had not ", text)
    text = re.sub(r"hadn't've", "had not have ", text)
    text = re.sub(r"hasn't", "has not ", text)
    text = re.sub(r"haven't", "have not ", text)
    text = re.sub(r"he'd", "he would ", text)
    text = re.sub(r"he'd've", "he would have ", text)
    text = re.sub(r"he'll", "he will ", text)
    text = re.sub(r"he'll've", "he he will have ", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"how'd", "how did ", text)
    text = re.sub(r"how'd'y", "how do you ", text)
    text = re.sub(r"how'll", "how will ", text)
    text = re.sub(r"how's", "how is ", text)
    text = re.sub(r"I'd", "I would ", text)
    text = re.sub(r"I'd've", "I would have ", text)
    text = re.sub(r"I'll", "I will ", text)
    text = re.sub(r"I'll've", "I will have ", text)
    text = re.sub(r"I'm", "I am ", text)
    text = re.sub(r"I've", "I have ", text)
    text = re.sub(r"i'd", "i would ", text)
    text = re.sub(r"i'd've", "i would have ", text)
    text = re.sub(r"i'll", "i will ", text)
    text = re.sub(r"i'll've", "i will have ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i've", "i have ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"it'd", "it would ", text)
    text = re.sub(r"it'd've", "it would have ", text)
    text = re.sub(r"it'll", "it will ", text)
    text = re.sub(r"it'll've", "it will have ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"let's", "let us ", text)
    text = re.sub(r"ma'am", "madam ", text)
    text = re.sub(r"mayn't", "may not ", text)
    text = re.sub(r"might've", "might have ", text)
    text = re.sub(r"mightn't", "might not ", text)
    text = re.sub(r"mightn't've", "might not have ", text)
    text = re.sub(r"must've", "must have ", text)
    text = re.sub(r"mustn't", "must not ", text)
    text = re.sub(r"mustn't've", "must not have ", text)
    text = re.sub(r"needn't", "need not ", text)
    text = re.sub(r"needn't've", "need not have ", text)
    text = re.sub(r"o'clock", "of the clock ", text)
    text = re.sub(r"oughtn't", "ought not ", text)
    text = re.sub(r"oughtn't've", "ought not hav ", text)
    text = re.sub(r"shan't", "shall not ", text)
    text = re.sub(r"sha'n't", "shall not ", text)
    text = re.sub(r"shan't've", "shall not have ", text)
    text = re.sub(r"she'd", "she would ", text)
    text = re.sub(r"she'd've", "she would have ", text)
    text = re.sub(r"she'll", "she will ", text)
    text = re.sub(r"she'll've", "she will have ", text)
    text = re.sub(r"she's", "she is ", text)
    text = re.sub(r"should've", "should have ", text)
    text = re.sub(r"shouldn't", "should not ", text)
    text = re.sub(r"shouldn't've", "should not have ", text)
    text = re.sub(r"so've", "so have ", text)
    text = re.sub(r"so have", "so as ", text)
    text = re.sub(r"that'd", "that would ", text)
    text = re.sub(r"that'd've", "that would have ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there'd", "there would ", text)
    text = re.sub(r"there'd've", "there would have ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"they'd", "they would ", text)
    text = re.sub(r"they'd've", "they would have ", text)
    text = re.sub(r"they'll", "they will ", text)
    text = re.sub(r"they'll've", "they will have ", text)
    text = re.sub(r"they're", "they are ", text)
    text = re.sub(r"they've", "they have ", text)
    text = re.sub(r"to've", "to have ", text)
    text = re.sub(r"wasn't", "was not ", text)
    text = re.sub(r"we'd", "we would ", text)
    text = re.sub(r"we'd've", "we would have ", text)
    text = re.sub(r"we'll", "we will ", text)
    text = re.sub(r"we'll've", "we will have ", text)
    text = re.sub(r"we're", "we are ", text)
    text = re.sub(r"we've", "we have ", text)
    text = re.sub(r"weren't", "were not ", text)
    text = re.sub(r"what'll", "what will ", text)
    text = re.sub(r"what'll've", "what will have ", text)
    text = re.sub(r"what're", "what are ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"when's", "when is ", text)
    text = re.sub(r"when've", "what have ", text)
    text = re.sub(r"where'd", "where did ", text)
    text = re.sub(r"where's", "where is ", text)
    text = re.sub(r"where've", "where have ", text)
    text = re.sub(r"who'll", "who will ", text)
    text = re.sub(r"who'll've", "who will have ", text)
    text = re.sub(r"who's", "who is ", text)
    text = re.sub(r"who've", "who have ", text)
    text = re.sub(r"why's", "why is ", text)
    text = re.sub(r"why've", "why have ", text)
    text = re.sub(r"will've", "will have ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"won't've", "will not have ", text)
    text = re.sub(r"would've", "would have ", text)
    text = re.sub(r"wouldn't", "would not ", text)
    text = re.sub(r"wouldn't've", "would not have ", text)
    text = re.sub(r"y'all", "you all ", text)
    text = re.sub(r"y'all'd", "you all would ", text)
    text = re.sub(r"y'all'd've", "you all would have ", text)
    text = re.sub(r"y'all're", "you all are ", text)
    text = re.sub(r"y'all've", "you all have ", text)
    text = re.sub(r"you'd", "you would ", text)
    text = re.sub(r"you'd've", "you would have ", text)
    text = re.sub(r"you'll", "you will ", text)
    text = re.sub(r"you'll've", "you will have ", text)
    text = re.sub(r"you're", "you are ", text)
    text = re.sub(r"you've", "you have ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def clean_punct(text): 
    words=tokenizer.tokenize(text)
    punctuation_filtered = []
    punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    regex = re.compile('[%s]' % re.escape(punct))
    for w in words:
        if w in selected_tags:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

def remove_stopwords(text):
    stop_words = set(stopword_list)
    words = tokenizer.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def normalize_corpus(corpus):
    normalized_corpus = []
    for doc in corpus:
        # strip HTML
        doc = strip_html_tags(doc)
        
        # remove accented characters
        doc = remove_accented_chars(doc)
        
        # lowercase the text
        doc = doc.lower()
            
        # expand contraction
        doc = expand_contractions(doc)
        
        #clean punctuations
        doc = clean_punct(doc)
    
        # remove stopwords
        doc = remove_stopwords(doc)

        # lemmatize text
        doc = lemmatize_text(doc)
            
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_characters(doc)
            
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        normalized_corpus.append(doc)
    return normalized_corpus

print(f"[INFO]: text processing functions loaded in memory")


################################ CLEANING
print(f"[INFO]: sanitizing text")
df['Title'] = normalize_corpus(df['Title'])
df['Body'] = normalize_corpus(df['Body'])
print("[INFO]: performed complete text sanitization")


# ############################### PICKLE DUMPS
# with open(f"{PROCESSED_DATA_DIR}/X_train.pkl", 'wb') as f:
#     pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)

# with open(f"{PROCESSED_DATA_DIR}/X_test.pkl", 'wb') as f:
#     pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

# with open(f"{PROCESSED_DATA_DIR}/y_train.pkl", 'wb') as f:
#     pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

# with open(f"{PROCESSED_DATA_DIR}/y_test.pkl", 'wb') as f:
#     pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)

# with open(f"{PROCESSED_DATA_DIR}/y_classes.pkl", 'wb') as f:
#     pickle.dump(multilabel_binarizer.classes_, f, pickle.HIGHEST_PROTOCOL)

# print("[INFO]: exported [X_train, X_test, y_train, y_test] pickles to disc")

# df_selected_tags = pd.DataFrame(selected_tags)
# df_selected_tags.to_pickle(f"{PROCESSED_DATA_DIR}/df_selected_tags.pkl")

df.to_pickle(f"{PROCESSED_DATA_DIR}/df_eda.pkl")

print("[INFO]: pickles dumped")
print("[INFO]: EDA finished")
