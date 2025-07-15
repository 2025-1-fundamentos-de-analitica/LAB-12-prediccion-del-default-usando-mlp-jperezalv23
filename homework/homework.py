# flake8: noqa: E501
import os
import gzip
import json
import pickle

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV


def limpiar_dataset(df):
    df = df.copy()
    df.drop(columns='ID', inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.dropna(inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df


def dividir_dataset(df_train, df_test):
    X_train = df_train.drop(columns="default")
    y_train = df_train["default"]
    X_test = df_test.drop(columns="default")
    y_test = df_test["default"]
    return X_train, y_train, X_test, y_test


def crear_pipeline():
    columnas_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']
    columnas_numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocesamiento = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
        ("numeric", StandardScaler(), columnas_numericas)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocesamiento),
        ("select_features", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("mlp", MLPClassifier(max_iter=15000, random_state=17)),
    ])

    return pipeline


def configurar_gridsearch(pipe):
    grid = {
        'pca__n_components': [None],
        'select_features__k': [20],
        'mlp__hidden_layer_sizes': [(50, 30, 40, 60)],
        'mlp__alpha': [0.26],
        'mlp__learning_rate_init': [0.001],
    }
    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )


def evaluar(modelo, X, y, nombre_conjunto):
    predicciones = modelo.predict(X)
    return predicciones, {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": round(precision_score(y, predicciones), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, predicciones), 4),
        "recall": round(recall_score(y, predicciones), 4),
        "f1_score": round(f1_score(y, predicciones), 4)
    }


def generar_matriz_confusion(y_real, y_pred, conjunto):
    matriz = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": conjunto,
        "true_0": {"predicted_0": int(matriz[0][0]), "predicted_1": int(matriz[0][1])},
        "true_1": {"predicted_0": int(matriz[1][0]), "predicted_1": int(matriz[1][1])}
    }


def guardar_comprimido(obj, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, 'wb') as f:
        pickle.dump(obj, f)


def guardar_jsonl(registros, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as archivo:
        for registro in registros:
            archivo.write(json.dumps(registro) + '\n')


if __name__ == "__main__":
    datos_train = pd.read_csv("files/input/train_data.csv.zip")
    datos_test = pd.read_csv("files/input/test_data.csv.zip")

    datos_train = limpiar_dataset(datos_train)
    datos_test = limpiar_dataset(datos_test)

    X_train, y_train, X_test, y_test = dividir_dataset(datos_train, datos_test)

    pipeline = crear_pipeline()
    modelo = configurar_gridsearch(pipeline)
    modelo.fit(X_train, y_train)

    guardar_comprimido(modelo, 'files/models/model.pkl.gz')

    y_pred_train, metricas_train = evaluar(modelo, X_train, y_train, "train")
    y_pred_test, metricas_test = evaluar(modelo, X_test, y_test, "test")

    matriz_train = generar_matriz_confusion(y_train, y_pred_train, 'train')
    matriz_test = generar_matriz_confusion(y_test, y_pred_test, 'test')

    guardar_jsonl(
        [metricas_train, metricas_test, matriz_train, matriz_test],
        'files/output/metrics.json'
    )
