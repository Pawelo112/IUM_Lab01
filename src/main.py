import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Wczytanie zbioru danych
    iris = load_iris()

    # Konwersja na DataFrame
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    # Eksploracja zbioru danych
    print("=== PIERWSZE WIERSZE ===")
    print(df.head())

    print("\n=== ROZMIAR ZBIORU ===")
    print(df.shape)

    print("\n=== STATYSTYKI OPISOWE ===")
    print(df.describe())

    # X zawiera cechy opisujace kwiaty
    X = df.drop(columns=["target"])

    # y zawiera klasy kwiatow
    y = df["target"]

    # Podzial danych na 80% treningowych i 20% testowych
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Utworzenie klasyfikatora drzewa decyzyjnego
    model = DecisionTreeClassifier(
        random_state=42
    )

    # Trenowanie modelu na danych treningowych
    model.fit(X_train, y_train)

    print("\n=== PODZIAL DANYCH ===")
    print(f"Zbior treningowy: {X_train.shape}")
    print(f"Zbior testowy: {X_test.shape}")

    print("\nModel drzewa decyzyjnego zostal wytrenowany.")

    # Predykcja klas na zbiorze testowym
    y_pred = model.predict(X_test)

    # Obliczenie skutecznosci klasyfikacji
    accuracy = accuracy_score(y_test, y_pred)

    # Pelny raport klasyfikacji
    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0
    )

    # Wyswietlenie wynikow
    print("\n=== OCENA MODELU ===")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Zapis wynikow do pliku tekstowego
    with open("wyniki.txt", "w", encoding="utf-8") as file:
        file.write("WYNIKI KLASYFIKACJI - Zbiór IRIS\n")
        file.write("==========================\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report)

    print("\nWyniki zapisano w pliku wyniki.txt")


if __name__ == "__main__":
    main()
