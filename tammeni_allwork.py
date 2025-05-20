# -*- coding: utf-8 -*-
"""Tammeni-allWork.ipynb

Original file is located at
    https://colab.research.google.com/drive/1BeQCiNWh2oksQbiTJ1Cb2uDH0jOU6L_v

# Imports and installations
"""

#!pip install jais transformers datasets torch evaluate regex accelerate
! pip install xgboost
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

arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()
label_encoder = LabelEncoder()
AnxEncoder = LabelEncoder()
DepEncoder = LabelEncoder()
smote = SMOTE(random_state=42)
Sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
model = SentenceTransformer('all-MiniLM-L6-v2')

"""# Reading Data

1st step
"""

'''df = pd.read_csv("/content/DATA-FORM - Form Responses 1 (2).csv") # data before Augmentation
cols_to_drop = [0,3,4,5,6,7,-1] # dropped unrelated cols
df = df.drop(df.columns[cols_to_drop], axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.rename(columns={'اذا تم تشخبصك من قبل طبيب نفسي، ما هو تشخيصك؟': 'Diagnosis',
                    'ما هي فئتك العمرية': 'Age Group',
                    'ما هو جنسك': 'Gender'}, inplace=True)
df['Diagnosis'] = [D.split(",") if "," in D else [D]for D in list(df['Diagnosis'])]
unwanted_diagnosis = ['اضطراب ثنائي القطب (Bipolar disorder)','اضطراب الوسواس القهري (Obsessive compulsive disorder (OCD))','لم يتم تشخيصي باحد هذه الأمراض','مرض آخر']
df['Diagnosis'] = [

        list(set([item.strip() if item.strip() not in unwanted_diagnosis else
        "Another Disorder"
        for item in diagnosis]))
     for diagnosis in df['Diagnosis']
]

rep = {
    'الاكتئاب ( Depression)' : 'Depression',
    'اضطراب القلق (Anxiety Disorder)' : "Anxiety",
    'غير مصاب بأي مرض نفسي': 'Healthy'

}
df['Diagnosis'] = df['Diagnosis'].apply(
    lambda diagnosis: [rep.get(item, item) for item in diagnosis]
)
df['Diagnosis'] = df['Diagnosis'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)'''

def extra(df):
# ensure Diagnosis is string
  df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('')
  # no brackets
  df['Diagnosis'] = df['Diagnosis'].str.strip('[]')
  # rows with multiple labels
  multi_label_rows = df[df['Diagnosis'].str.contains(',')]
  # Split them
  df_multi = multi_label_rows.copy()
  df_multi['Diagnosis'] = df_multi['Diagnosis'].str.split(',')
  df_multi = df_multi.explode('Diagnosis')
  # Clean spaces
  df_multi['Diagnosis'] = df_multi['Diagnosis'].str.strip()
  # Get single-label rows
  df_single = df[~df.index.isin(multi_label_rows.index)]
  # Combine
  df_final = pd.concat([df_single, df_multi], ignore_index=True)
  df_final['Diagnosis'] = df_final['Diagnosis'].str.strip().str.strip("'")

  df_final = df_final[df_final['Diagnosis'].isin(['Depression', 'Anxiety', 'Healthy'])].reset_index(drop=True)
  return df_final

# df = extra(df)

"""Data Augmentation"""

'''
#the 6 columns which are question-answer fields
qa_columns = df.columns[2:8]
# STEP 2: Define synonym replacement dictionary
synonyms_dict = {
    "الحزن": ["الكآبة", "الأسى", "الغم"],
    "التعب": ["الإرهاق", "الإنهاك"],
    "القلق": ["الخوف", "التوتر", "الانزعاج"],
    "الارهاق": ["التعب", "الإنهاك", "الإجهاد"],
    "فقدان": ["غياب", "انعدام"],
    "الأمل": ["الرجاء", "التفاؤل"],
    "التركيز": ["الانتباه", "التركيز الذهني"],
    "الخوف": ["الرهبة", "القلق"],
    "النوم": ["السبات", "الراحة"],
    "افكر": ["اشغل ذهني", "اكرر التفكير"],
    "اشعر": ["احس", "اواجه شعورًا بـ"],
    "عصبية": ["توتر", "انفعال"],
    "التفكير": ["الانشغال الذهني", "التأمل القلق"],
    "نعم": ["بالفعل", "أجل"],
    "لا": ["لم يحدث", "غير صحيح"],
    "صعوبة": ["مشكلة", "تحدي", "معاناة"],
    "اريد": ["أرغب", "أتمنى"],
    "اعاني": ["اشكو", "اواجه"],
    "خارج عن ارادتي": ["غير قابل للسيطرة", "لا أتحكم فيه"],
    "ضيق": ["ضغط", "توتر شديد"],
    "حياتي": ["روتيني", "نشاطي اليومي"],
}

# Synonym replacement function
def replace_with_synonyms(text, synonyms_dict):
    if not isinstance(text, str):
        return text
    for word, synonym_list in synonyms_dict.items():
        pattern = rf'\b{word}\b'
        if re.search(pattern, text):
            synonym = random.choice(synonym_list)
            text = re.sub(pattern, synonym, text)
    return text

# Generate augmented answers
df_augmented = df.copy()
for col in qa_columns:
    new_col = "aug_" + col[:20].strip().replace("\n", "")
    df_augmented[new_col] = df[col].apply(lambda x: replace_with_synonyms(x, synonyms_dict))

#  Define and apply long replacements for placeholder answers
short_phrases = [
    "حصل", "نعم", "غير صحيح", "جميع ما ذكر", "لا", "الحمد لله",
    "كل الاعراض", "لم يحدث", "للاسف"
]

long_replacements = [
    "عانيت من العديد من الأعراض التي أثرت على حياتي بشكل كبير، من بينها الحزن المستمر والتعب الذهني.",
    "نعم، كنت أعاني من أعراض واضحة ومؤثرة، وقد استمرت لفترة طويلة وأثرت على يومي بالكامل.",
    "إجابتي السابقة لم تكن دقيقة، أحتاج لتوضيح أنني مررت بأعراض نفسية حادة استدعت المساعدة.",
    "تعرضت فعلاً لكل ما تم ذكره، لكن تأثيرها اختلف من وقت لآخر وكان يتفاقم أحيانًا.",
    "في بعض الأحيان لم ألاحظ الأعراض بوضوح، لكن بعد التفكير اكتشفت أنها كانت تؤثر على تركيزي.",
    "لم تكن الأعراض بسبب أسباب عضوية أو دوائية بل نتيجة ضغوط نفسية متراكمة.",
    "جميع الأعراض كانت موجودة تقريبًا، وخصوصًا فقدان الشغف والتوتر وصعوبات النوم.",
    "لا أذكر حالة طبية أو دوائية محددة، كانت حالتي مرتبطة أساسًا بالمشاعر والسلوك اليومي.",
    "كنت أشعر أن هذه الأعراض خارجة عن إرادتي وأثرت على تفاعلي مع الآخرين بشكل سلبي."
]

def replace_short_answers(text):
    if not isinstance(text, str):
        return text
    if text.strip() in short_phrases:
        return random.choice(long_replacements)
    return text

# Apply to augmented QA columns
augmented_columns = [col for col in df_augmented.columns if col.startswith("aug_")]
for col in augmented_columns:
    df_augmented[col] = df_augmented[col].apply(replace_short_answers)

# Copy Age, Gender, Diagnosis to augmented version
df_augmented_final = df_augmented[augmented_columns].copy()
df_augmented_final["Age Group"] = df["Age Group"]
df_augmented_final["Gender"] = df["Gender"]
df_augmented_final["Diagnosis"] = df["Diagnosis"]

# Rename augmented columns to match original
df_augmented_final.columns = qa_columns.tolist() + ["Age Group", "Gender", "Diagnosis"]

# Add Source column and combine both
df_augmented_final["Source"] = "augmented"
df_original_final = df[qa_columns.tolist() + ["Age Group", "Gender", "Diagnosis"]].copy()
df_original_final["Source"] = "original"

combined_df = pd.concat([df_original_final, df_augmented_final], ignore_index=True)
final_df = combined_df.drop_duplicates(subset=qa_columns.tolist(), keep='first')
final_df.to_excel("final_tammen.xlsx", index=False)
print("Saved to: final_tammen.xlsx")'''

"""# Data Cleaning

For JAIS
"""

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

"""# JAIS"""

# !accelerate config

'''login(token="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")
# 8 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("inceptionai/jais-13b-chat", padding_side='left')

model = AutoModelForCausalLM.from_pretrained(
    "inceptionai/jais-13b-chat",
    device_map="auto",
    trust_remote_code=True
)'''

'''def translate_to_fusha(text):
    try:
        # Skip empty/invalid strings
        if not isinstance(text, str) or text.strip() == "":
            return ""

        prompt = f"""حول الجملة التالية من اللهجة العامية إلى اللغة العربية الفصحى:\n"{text}"\nالنتيجة:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleaning
        del inputs, outputs
        torch.cuda.empty_cache()

        return response.split("النتيجة:")[-1].strip()

    except Exception as e:
        print(f"Error translating: {text[:30]}... — {e}")
        torch.cuda.empty_cache()
        return ""'''

'''translated_df = pd.DataFrame(index=df.index, columns=df.columns[2:-1])
for i, row in enumerate(df.index):
    for col in df.columns[2:8]:
        output = translate_to_fusha(df.at[row, col])
        print(output)
        translated_df.at[row, col] = output'''

"""# Data Exploration"""

'''# token/word count per combined response
text_columns = df.columns.to_list()[:6]
df['combined_text'] = df[text_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
df['word_count'] = df['combined_text'].apply(lambda x: len(nltk.word_tokenize(x)))

# distribution of response lengths
plt.figure(figsize=(10, 4))
sns.histplot(df['word_count'], bins=30, kde=True)
plt.title('Distribution of Word Counts per Entry')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()'''

"""# Data Preprocessing"""

def clean_and_stem_arabic(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in arabic_stopwords]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

"""# MODELS

### Dynamic Similarity
"""

def cosine_similarity_positive_only(vec1, vec2):
    if hasattr(vec1, 'cpu'):
        vec1 = vec1.cpu().numpy()
    if hasattr(vec2, 'cpu'):
        vec2 = vec2.cpu().numpy()
    # mask
    mask = (vec1 != 0) & (vec2 != 0)
    if np.sum(mask) == 0:
        return 0.0
    v1 = vec1[mask]
    v2 = vec2[mask]
    # cosine similarity
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(sim)

"""## SMOTE"""

def up_sample(X, Y):
  return smote.fit_resample(X, Y)

"""## SBERT"""

def encode_Sbert(questions,answers):
  questions = [clean_text(text) for text in questions]
  #questions = [clean_and_stem_arabic(text) for text in questions]
  question_embeddings = Sbert.encode(questions, convert_to_tensor=True,normalize_embeddings=True)
  similarities = []
  for _, answer in answers.iterrows():
    answer = answer.tolist()
    answer_embedding = Sbert.encode([answer], convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity to all question embeddings
    row_similarities = cos_sim(answer_embedding, question_embeddings)[0]  # returns 2D array
    similarities.append(row_similarities)
  df = pd.DataFrame(similarities, columns=[f"Q{i+1}_sim" for i in range(len(questions))])
  return df

"""## Grid Search"""

def Grid_search(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  param_grid = {
    'n_estimators': [50,100,150, 200, 300],
    'max_depth': [None, 5,10, 20, 30],
    'min_samples_split': [2, 4, 5, 6, 10],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_leaf_nodes': [None, 10, 20],
    'max_features': ['sqrt', 'log2', 0.6, 0.4, None],
    'bootstrap': [True],
    'max_samples': [None, 0.5,0.6,0.7, 0.8],
    'class_weight': [None, 'balanced'],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0.0, 0.01, 0.02, 0.05]
  }
  rfc = RandomForestClassifier(random_state=42)
  random_search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_grid,
    n_iter=150,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
  )
  random_search.fit(X_train, y_train)
  return random_search.best_params_

from sklearn.model_selection import GridSearchCV

def svm_grid_search(X_train, y_train, X_test, y_test, param_grid=None, cv=5, scoring='accuracy'):

    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    y_pred = grid_search.predict(X_test)
    test_report = classification_report(y_test, y_pred)

    return best_params

"""## Consfusion Matrix"""

def ConfusionMatrix(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.show()

"""## Learning Curve"""

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

"""## Models

getting probablilty
"""

def get_score(model, X_test):
  return model.predict_proba(X_test)

"""### Random Forest Classifier"""

def RFC(X,y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

  rfc = RandomForestClassifier(**Grid_search(X,y))
  rfc.fit(X_train, y_train)
  y_pred = rfc.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(rfc, X_train, y_train)
  return rfc, get_score(rfc, X_test)

"""### Decisoin Tree"""

def DT(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
   )
  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  print("Classification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(dt, X_train, y_train)
  return dt, get_score(dt, X_test)

"""### XGBoost"""

def XGBoost(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  XGB = xgb.XGBClassifier(eval_metric='mlogloss')
  XGB.fit(X_train, y_train)
  y_pred = XGB.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  print("Classification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(XGB, X_train, y_train)
  return XGB, get_score(XGB, X_test)

"""### Logistic Regression

LR with elasticnet
"""

def LR_ENET(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  LR = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,         # 0.0 = Ridge, 1.0 = Lasso, in between = Elastic Net
    C=1.0,                # Inverse of regularization strength (like 1/alpha)
    max_iter=1000,
    random_state=42
)

  LR.fit(X_train, y_train)
  y_pred = LR.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print(f"Accuracy: {accuracy}")
  print("Classification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(LR, X_train, y_train)
  return LR, get_score(LR, X_test)

"""LR with lasso"""

def LR_LASSO(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # L1 (Lasso-like) logistic regularization
  LR = LogisticRegression(
      penalty='l1',
      solver='liblinear',
      C=1.0,
      random_state=42
  )
  LR.fit(X_train, y_train)
  y_pred = LR.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  print("Classification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(LR, X_train, y_train)
  return LR, get_score(LR, X_test)

"""### Support Vector Machine"""

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

"""### K Nearest Neighbours"""

def KNN(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}%")
  print("\nClassification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(knn, X_train, y_train)
  return knn, get_score(knn, X_test)

"""### Naive Bayes"""

def NB(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  nb = GaussianNB(priors=None, var_smoothing=1e-9)
  nb.fit(X_train,y_train)
  y_pred = nb.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}%")
  print("\nClassification Report:\n", classification_report(y_test, y_pred, digits = 4))
  ConfusionMatrix(y_test, y_pred)
  LearningCurve(nb, X_train, y_train)
  return nb, get_score(nb, X_test)

"""# Question Answer Embedding Approach"""

def QA(questions, input):
  questions = [clean_text(text) for text in questions]
  input.columns = questions

  all_embeddings = []

  for idx, row in input.iterrows():
      answers = [row[q] for q in questions]
      qa_pairs = [(q, a) for q, a in zip(questions, answers)]

      texts_to_encode = []
      for q, a in qa_pairs:
          texts_to_encode.extend([q, a])

      embeddings = Sbert.encode(texts_to_encode)
      sample_embedding = np.concatenate(embeddings).flatten()
      all_embeddings.append(sample_embedding)
  embds = np.array(all_embeddings)
  return embds
# all_embeddings = all_embeddings.flatten()
# pca = PCA(n_components=2)
# reduced_vector = pca.fit_transform(embds)

"""# TF-IDF Instaed of SBert"""

def encode_tfidf(questions, answers):
    questions = [clean_text(text) for text in questions]
    vectorizer = TfidfVectorizer()
    all_texts = questions + answers.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    vectorizer.fit(all_texts)
    question_tfidf = vectorizer.transform(questions)
    similarities = []
    for _, answer in answers.iterrows():
        answer_text = ' '.join(answer.astype(str))
        answer_tfidf = vectorizer.transform([answer_text])
        row_similarities = cosine_similarity(answer_tfidf, question_tfidf)[0]
        similarities.append(row_similarities)

    df = pd.DataFrame(similarities, columns=[f"Q{i+1}_sim" for i in range(len(questions))])
    return df

"""# Dividing Depression and Anxiety

"""

df = pd.read_excel("/content/final_tammeni_augmented_final.xlsx")
df.iloc[::,:6] = df.iloc[::,:6].astype(str).applymap(clean_text)

df

#  depression
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

"""## Cosine Similairity and SBert for Depression

# SVM

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
svm_dep, SVMDepScore = SVM(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
svm_anx, SVMAnxScore = SVM(X,y)

# import joblib
# joblib.dump(svm_dep, "SVM_DEPRESSION_last.pkl", compress=3, protocol=4)
# joblib.dump(svm_anx, "SVM_ANXIETY_last.pkl", compress=3, protocol=4)

"""# RFC

##  Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
rfc_dep, RFCDepScore = RFC(X,y)
# print(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
rfc_anx, RFCAnxScore = RFC(X,y)

"""# XGBoost

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
xgb_dep , xgbScore= XGBoost(X,y)
# print(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
xgb_anx, XGBAnxScore = XGBoost(X,y)

"""# DT

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
DT_dep , DTScore= DT(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
dt_anx, DTAnxScore = DT(X,y)

"""# LR LASSO

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
LRL_dep , LRLScore= LR_LASSO(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
LRL_anx, LRLAnxScore = LR_LASSO(X,y)

"""# LR_ENET

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
LRE_dep , LREScore= LR_ENET(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
LRE_anx, LREAnxScore = LR_ENET(X,y)

"""# KNN

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
KNN_dep , KNNScore= KNN(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
KNN_anx, KNNAnxScore = KNN(X,y)

"""# Naive BAYES

## Depression
"""

questions_dep = df_dep.columns.to_list()[:3]
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X, y = up_sample(X, y)
NB_dep , NBScore= NB(X,y)

"""## Anxiety"""

questions_anx = df_anx.columns.to_list()[:4]
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X, y = up_sample(X, y)
NB_anx, NBAnxScore = NB(X,y)

"""# Question Answer Embdeiing approach

## Depression
"""

y = df_dep['Diagnosis']
X = QA(questions_dep, answers_dep)
X, y = up_sample(X, y)
svm_dep, SVMDepScore = SVM(X,y)

"""## Anxiety"""

y = df_anx['Diagnosis']
X, y = up_sample(QA(questions_anx, answers_anx), y)
svm_anx, SVMAnxScore = SVM(X,y)

"""# Sbert with sentiment analysis"""

from transformers import pipeline
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

label_map = {
    'negative': -1,
    'neutral': 0,
    'positive': 1
}

questions_dep = df_dep.columns.to_list()[:3]
df_dep['SA_text'] = df_dep[questions_dep[0]] + ' ' + df_dep[questions_dep[1]] + ' ' + df_dep[questions_dep[2]]
df_dep['SA_class'] = df_dep['SA_text'].apply(lambda x: classifier(str(x))[0]['label'])
df_dep['SA_class'] = df_dep['SA_class'].map(label_map)
df_dep.drop(['SA_text'], axis=1, inplace=True)
answers_dep = df_dep[questions_dep]
X = encode_Sbert(questions_dep, answers_dep)
y = df_dep['Diagnosis']
X.reset_index(drop=True, inplace=True)
df_dep.reset_index(drop=True, inplace=True)
X['class'] = df_dep['class']

X, y = up_sample(X, y)
svm_dep, SVMDepScore = SVM(X,y)
print(X,y)

questions_anx = df_anx.columns.to_list()[:4]
df_anx['SA'] = df_anx[questions_anx[0]].astype(str) + ' ' + df_anx[questions_anx[1]].astype(str) + ' ' + df_anx[questions_anx[2]].astype(str)+' '+ df_anx[questions_anx[3]].astype(str)
df_anx['class'] = df_anx['SA'].apply(lambda x: classifier(x)[0]['label'])
df_anx.drop(['SA'], axis= 1, inplace = True)
df_anx['class'] = df_anx['class'].map(label_map)
answers_anx = df_anx[questions_anx]
X = encode_Sbert(questions_anx, answers_anx)
y = df_anx['Diagnosis']
X.reset_index(drop=True, inplace=True)
df_anx.reset_index(drop=True, inplace=True)
X['class'] = df_anx['class']

X, y = up_sample(X, y)
svm_anx, svmAnxScore = SVM(X,y)
print(X,y)
