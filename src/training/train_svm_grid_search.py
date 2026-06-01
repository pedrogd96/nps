import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def main():
    data = pd.read_csv("data/processed/nps_training_vectorized.csv")
    yColumn = "responsavel"
    RANDOM_STATE = 42

    X = data.drop(yColumn, axis=1)
    y = data[yColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ("svc", SVC(random_state=RANDOM_STATE))
    ])

    params_grid = {
        'svc__kernel': ['sigmoid'],
        'svc__C': np.random.uniform(4, 45, 100),
        'svc__class_weight': [None]
    }

    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params_grid,
        scoring='f1_weighted',
        cv=splitter,
        refit=True,
        verbose=10,
        error_score=0
    )
        
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    yhat_test = grid_search.best_estimator_.predict(X_test)

    print('Desempenho - Base de Teste')
    print(classification_report(y_test, yhat_test))
    print('Matriz de confusão - Base de Teste')
    print(confusion_matrix(y_test, yhat_test))


if __name__ == "__main__":
    main()