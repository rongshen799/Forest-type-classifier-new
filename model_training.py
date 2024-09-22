from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from joblib import dump, load
import os

from data_preparation import prepare_data  
from data_preparation import augment_data  


def train_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
    }

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} Cross-validation Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        model.fit(X_train, y_train)
        models[name] = model

    # Create and train voting classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    models['VotingClassifier'] = voting_clf

    return models

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


def save_models(models, compress_level=9):
    for name, model in models.items():
        filename = f'{name}_model_compressed.joblib'
        dump(model, filename, compress=compress_level)
        print(f"Saved {name} model. Size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

def load_models(model_names):
    models = {}
    for name in model_names:
        filename = f'{name}_model_compressed.joblib'
        models[name] = load(filename)
    return models

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, poly = prepare_data()
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    models = train_models(X_train_aug, y_train_aug)
    dump(poly, 'poly_compressed.joblib', compress=9)
    evaluate_models(models, X_test, y_test)
    
    save_models(models)

    loaded_models = load_models(['RandomForest', 'ExtraTrees', 'LightGBM', 'CatBoost', 'VotingClassifier'])
    evaluate_models(loaded_models, X_test, y_test)

    for name in models.keys():
        filename = f'{name}_model_compressed.joblib'
        print(f"{name} model size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")
