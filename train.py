"""
train.py - trains Naive Bayes, Logistic Regression and SVM on the
Fake and Real News dataset, compares them, and saves the best one.

Run:  python train.py
Needs: data/Fake.csv and data/True.csv
"""
import re
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# 1. LOAD DATA -----------------------------------------------------------
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")
fake["label"] = 0   # 0 = Fake
real["label"] = 1   # 1 = Real
df = pd.concat([fake, real], ignore_index=True)

# 2. CLEAN DATA ----------------------------------------------------------
# Join headline + body into one text column
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# The real articles start with things like "WASHINGTON (Reuters) -".
# If we leave it, the model just learns "Reuters = real" (cheating).
# So we remove it. We also do NOT use the 'subject' column for the same reason.
df["content"] = df["content"].apply(
    lambda t: re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", t)
)
df = df[df["content"].str.strip().str.len() > 20]
df = df.drop_duplicates(subset="content")
print(f"Articles after cleaning: {len(df)}  (fake={sum(df.label==0)}, real={sum(df.label==1)})")

# 3. SPLIT 80/20 ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 4. MODELS --------------------------------------------------------------
# TfidfVectorizer already does: lowercase, remove punctuation (tokenizing),
# stop-word removal, and TF-IDF in one step.
def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7,
                                  max_features=50000, ngram_range=(1, 2))),
        ("clf", classifier),
    ])

models = {
    "Naive Bayes": make_pipeline(MultinomialNB()),
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
    # Calibrated so the SVM can output a confidence score too
    "SVM": make_pipeline(CalibratedClassifierCV(LinearSVC())),
}

# 5. TRAIN + EVALUATE ----------------------------------------------------
results, trained = [], {}
for name, pipe in models.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
    })
    trained[name] = pipe

results_df = pd.DataFrame(results).round(4)
print("\n=== RESULTS ===")
print(results_df.to_string(index=False))

# 6. SAVE BEST MODEL (by F1) --------------------------------------------
best_name = results_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
print(f"\nBest model: {best_name}")
joblib.dump(trained[best_name], "model.joblib")
results_df.to_csv("results.csv", index=False)
with open("best_model.txt", "w") as f:
    f.write(best_name)
print("Saved model.joblib, results.csv, best_model.txt")
