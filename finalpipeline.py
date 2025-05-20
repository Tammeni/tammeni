# -*- coding: utf-8 -*-
"""FinalPipeline.ipynb

Original file is located at
    https://colab.research.google.com/drive/1UpocnXIBOOxT40hdVl3qAaJuDEtiJhw_
"""

#!pip install jais transformers datasets torch evaluate regex accelerate xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from huggingface_hub import login

import regex as reg
import re
import ast
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')



label_encoder = LabelEncoder()
AnxEncoder = LabelEncoder()
DepEncoder = LabelEncoder()
smote = SMOTE(random_state=42)
Sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

def clean_text(text):
  cleaned = re.sub(r'[\'\"\n\d,;.،؛.؟{}():]', ' ', text)
  cleaned = re.sub(r'\s{2,}', ' ', cleaned)

  emoji_pattern = re.compile("["
  u"\U0001F600-\U0001F64F" # emoticons
  u"\U0001F300-\U0001F5FF" # symbols
  u"\U0001F680-\U0001F6FF" # transport & map symbols
  u"\U0001F1E0-\U0001F1FF" # flags
  u"\U00002702-\U000027B0"
  u"\U000024C2-\U0001F251"
  "]+", flags=re.UNICODE)
  cleaned = emoji_pattern.sub(r'', cleaned)
  cleaned = re.sub(r'[\u064B-\u0652]', '', cleaned) # tashkeel removal
  # Normalize -> Alif
  cleaned = re.sub(r'[إأآا]', 'ا', cleaned)
  # Normalization -> Ta Marbuta to Ha
  cleaned = cleaned.replace('ة', 'ه')
  # Normalization -> Yeh
  cleaned = cleaned.replace('ى', 'ي')
  cleaned = cleaned.replace("ؤ", "و")
  cleaned = cleaned.replace("ئ", "ي")
  cleaned = re.sub(r'[^\u0600-\u06FF\s]', '', cleaned) # punctuation and latin characters removal
  return cleaned.strip()

def up_sample(X, Y):
  return smote.fit_resample(X, Y)

def encode_Sbert(questions,answers):
  questions = [clean_text(text) for text in questions]
  #questions = [clean_and_stem_arabic(text) for text in questions]
  question_embeddings = Sbert.encode(questions, convert_to_tensor=True,normalize_embeddings=True)
  similarities = []
  for _, answer in answers.iterrows():
    answer = answer.tolist()
    answer_embedding = Sbert.encode([answer], convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity to all question embeddings
    row_similarities = cosine_similarity(answer_embedding, question_embeddings)[0]  # returns 2D array
    similarities.append(row_similarities)
  df = pd.DataFrame(similarities, columns=[f"Q{i+1}_sim" for i in range(len(questions))])
  return df

def ConfusionMatrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

def LearningCurve(estimator, X, y, title=None):

    fig, ax = plt.subplots(figsize=(7, 5))

    LearningCurveDisplay.from_estimator(
        estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        score_type="both",
        n_jobs=4,
        line_kw={"marker": "o"},
        std_display_style="fill_between",
        score_name="Accuracy",
        ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(title or f"Learning Curve for {estimator.__class__.__name__}")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def get_score(model, X_test):
  return model.predict_proba(X_test)

def SVM(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42,probability=True)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)
  y_prob = svm.predict_proba(X_test)
  roc_auc = roc_auc_score(y_test, y_prob[:, 1])
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("ROC AUC Score:", round(roc_auc, 4))
  print("\nClassification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(svm, X_train, y_train)
  svm_scores = get_score(svm, X_test)
  return svm, svm_scores

df = pd.read_excel("/content/final_tammeni_augmented_final.xlsx")
df.iloc[::,:6] = df.iloc[::,:6].astype(str).applymap(clean_text)

df = df[~df['Diagnosis'].apply(lambda x: "Another Disorder" in x)]
# depression
df_dep = df.iloc[::,[0,1,2,8]]
df_dep = df_dep[~df_dep['Diagnosis'].apply(lambda x: "Anxiety" in x)].reset_index(drop=True)
df_dep['Diagnosis'] = df_dep['Diagnosis'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
df_dep['Diagnosis'] = DepEncoder.fit_transform(df_dep['Diagnosis'])
# anxiety
df_anx = df.iloc[::,[2,3,4,5,8]]
df_anx = df_anx[~df_anx['Diagnosis'].apply(lambda x: "Depression" in x)].reset_index(drop=True)
df_anx['Diagnosis'] = df_anx['Diagnosis'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
df_anx = df_anx[~df_anx['Diagnosis'].apply(lambda x: 'Anxiety' in x and 'Healthy' in x)].reset_index(drop=True)
df_anx['Diagnosis'] = AnxEncoder.fit_transform(df_anx['Diagnosis'])

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
svm_dep, SVMDepScore = SVM(X,y)

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
svm_anx, SVMAnxScore = SVM(X,y)
