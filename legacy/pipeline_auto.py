import optuna
import logging
import sys

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, 
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Suppress the specific FutureWarning - only for seaborn
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning)

def data_cleansing():
    """ Data Prepare """
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(
                [
                    'Lady', 'Countess','Capt', 
                    'Col', 'Don', 'Dr', 
                    'Major', 'Rev', 'Sir', 
                    'Jonkheer', 'Dona'
                ], 'Rare'
            )
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)  
    
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1) 
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(2):
            for j in range(3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
                
        for i in range(2):
            for j in range(3):
                dataset.loc[
                    (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'
                ] = guess_ages[i, j]
        dataset['Age'] = dataset['Age'].astype(int)
    
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
        
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]
    
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
        
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    test_df.fillna({'Fare': test_df['Fare'].dropna().median()}, inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    return train_df, test_df
    
def objective(trial):
    train_df, test_df = data_cleansing()
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    # print(X_train.shape, Y_train.shape, X_test.shape)

    classifier_name = trial.suggest_categorical("classifier", [
        'Logistic Regression', 
        'Perceptron', 
        # 'Stochastic Gradient Decent', 
        # 'Support Vector Machines', 
        # 'Linear SVC',        
        # 'Random Forest',
        # 'AdaBoost', 
        # 'GradientBoosting',
        # 'ExtraTrees',
        # 'KNN',  
        # 'Gaussian Naive Bayes', 
        # 'Decision Tree', 
    ])
    if classifier_name == "Logistic Regression":
        tol = trial.suggest_float("tol", 1e-10, 1e-1, log=True)
        C = trial.suggest_float("C", 1e-10, 1, log=True)
        solver = trial.suggest_categorical("solver", ['lbfgs', 'liblinear', 'newton-cholesky'])
        max_iter = trial.suggest_int("max_iter", 1e2, 1e5, log=True)
        classifier_obj = LogisticRegression(tol=tol, C=C, solver=solver, max_iter=max_iter)
    elif classifier_name == "Perceptron":
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", 'elasticnet'])
        alpha = trial.suggest_float("alpha", 1e-10, 1e-2, log=True) 
        max_iter = trial.suggest_int("max_iter", 1e2, 1e5, log=True)
        tol = trial.suggest_float("tol", 1e-10, 1e-1, log=True)
        early_stopping = trial.suggest_categorical("early_stopping", [True, False])
        classifier_obj = Perceptron(penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, early_stopping=early_stopping)
    # elif classifier_name == "Stochastic Gradient Decent":
    #     classifier_obj = SGDClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "Support Vector Machines":
    #     classifier_obj = SVC(C=svc_c, gamma="auto")
    # elif classifier_name == "Linear SVC":
    #     classifier_obj = LinearSVC(C=svc_c, gamma="auto")
    # elif classifier_name == "Random Forest":
    #     classifier_obj = RandomForestClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "AdaBoost":
    #     classifier_obj = AdaBoostClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "GradientBoosting":
    #     classifier_obj = GradientBoostingClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "ExtraTrees":
    #     classifier_obj = ExtraTreesClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "KNN":
    #     classifier_obj = KNeighborsClassifier(C=svc_c, gamma="auto")
    # elif classifier_name == "Gaussian Naive Bayes":
    #     classifier_obj = GaussianNB(C=svc_c, gamma="auto")
    # elif classifier_name == "Decision Tree":
    #     classifier_obj = DecisionTreeClassifier(C=svc_c, gamma="auto")

    score = cross_val_score(classifier_obj, X_train, Y_train, n_jobs=-1, cv=5, scoring="accuracy")
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    optuna.logging.get_logger("titanic").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "titanic-study"  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    # print(trial)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# if __name__ == "__main__":
#     clf = SVC(gamma="auto")

#     param_distributions = {
#         "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
#         "degree": optuna.distributions.IntDistribution(1, 5),
#     }

#     optuna_search = optuna.integration.OptunaSearchCV(
#         clf, param_distributions, n_trials=100, timeout=600, verbose=0
#     )

#     X, y = load_iris(return_X_y=True)
#     optuna_search.fit(X, y)

#     print("Best trial:")
#     trial = optuna_search.study_.best_trial

#     print("  Value: ", trial.value)
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))