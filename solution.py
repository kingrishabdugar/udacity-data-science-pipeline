"""Train a mixed numeric, categorical, and text model pipeline."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def build_pipeline():
    """Return a complete preprocessing and classification pipeline."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    text = Pipeline([("flatten", FunctionTransformer(lambda x: x.astype(str).values.ravel(), validate=False)), ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_features=12000))])
    prep = ColumnTransformer([("numeric", numeric, ["Age", "Positive Feedback Count", "Review Length"]), ("categorical", categorical, ["Division Name", "Department Name", "Class Name"]), ("review", text, "Review Text"), ("title", text, "Title")])
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear"))])


def train_and_evaluate(data_path="data/reviews.csv"):
    """Train with cross-validated tuning and return model plus test metrics."""
    data = pd.read_csv(data_path)
    data["Review Text"] = data["Review Text"].fillna("")
    data["Title"] = data["Title"].fillna("")
    data["Review Length"] = data["Review Text"].str.len()
    features = ["Review Text", "Title", "Age", "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "Review Length"]
    X, y = data[features], data["Recommended IND"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    search = GridSearchCV(build_pipeline(), {"model__C": [.5, 1.0]}, cv=3, scoring="f1", n_jobs=1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "best_params": search.best_params_}
    return search.best_estimator_, metrics


if __name__ == "__main__":
    model, metrics = train_and_evaluate()
    print(metrics)
