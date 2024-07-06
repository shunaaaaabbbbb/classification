import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

def load_and_split_data(data_path):
    """データの読み込みと訓練データとテストデータへの分割を行う
    (変更なし)
    """
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def lda_classification(X_train, X_test, y_train, y_test):
    """線形判別分析を用いた分類"""
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_test, y_pred

def decision_tree_classification(X_train, X_test, y_train, y_test, **params):
    """決定木を用いた分類"""
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_test, y_pred

def random_forest_classification(X_train, X_test, y_train, y_test, **params):
    """ランダムフォレストを用いた分類"""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_test, y_pred

def knn_classification(X_train, X_test, y_train, y_test):
    """k近傍法を用いた分類"""
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_test, y_pred

def lightgbm_classification(X_train, X_test, y_train, y_test, **params):
    """LightGBMを用いた分類"""
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_test, y_pred