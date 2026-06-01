import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv("data/processed/nps_training_vectorized.csv")
    yColumn = "responsavel"
    RANDOM_STATE = 42

    X = df.drop(columns=[yColumn])
    y = df[yColumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nCLASSIFICATION REPORT\n")
    print(classification_report(y_test, predictions))
    print("\nCONFUSION MATRIX\n")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()