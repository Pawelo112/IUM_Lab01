import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Wyznaczenie katalogu glownego projektu
BASE_DIR = Path(__file__).resolve().parent.parent

# Wczytanie lokalnej konfiguracji z pliku .env
load_dotenv(BASE_DIR / ".env")

# Pobranie parametrow konfiguracyjnych
DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def load_dataset():
    """Wczytuje dane Iris i zapisuje je do CSV, jesli plik nie istnieje."""

    # Wczytanie gotowego zbioru z biblioteki scikit-learn
    iris = load_iris()

    # Konwersja tablicy cech na pandas.DataFrame
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    # Dodanie kolumny zawierajacej etykiety klas
    df["target"] = iris.target

    # Utworzenie katalogu danych, jesli nie istnieje
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Zapis danych do CSV przy pierwszym uruchomieniu
    if not DATA_PATH.exists():
        df.to_csv(DATA_PATH, index=False)
        print(f"Utworzono plik danych: {DATA_PATH}")

    # Odczyt zbioru ze sciezki wskazanej w konfiguracji
    df = pd.read_csv(DATA_PATH)

    return df, iris.target_names


def main():
    # Wczytanie danych i nazw klas
    df, class_names = load_dataset()

    # Zadanie 3: eksploracja danych
    print("=== PIERWSZE WIERSZE ===")
    print(df.head())

    print("\n=== ROZMIAR ZBIORU ===")
    print(df.shape)

    print("\n=== STATYSTYKI OPISOWE ===")
    print(df.describe())

    # Zadanie 4: przygotowanie cech i etykiet
    X = df.drop(columns=["target"])
    y = df["target"]

    # Podzial na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Utworzenie modelu drzewa decyzyjnego
    model = DecisionTreeClassifier(
        random_state=RANDOM_SEED
    )

    # Trenowanie klasyfikatora
    model.fit(X_train, y_train)

    print("\n=== PODZIAL DANYCH ===")
    print(f"Zbior treningowy: {X_train.shape}")
    print(f"Zbior testowy: {X_test.shape}")
    print("Model drzewa decyzyjnego zostal wytrenowany.")

    # Zadanie 5: predykcja na zbiorze testowym
    y_pred = model.predict(X_test)

    # Obliczenie metryki accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Utworzenie szczegolowego raportu klasyfikacji
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\n=== OCENA MODELU ===")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Zapis wynikow do pliku tekstowego
    results_path = BASE_DIR / "wyniki.txt"

    with open(results_path, "w", encoding="utf-8") as file:
        file.write("WYNIKI KLASYFIKACJI - Zbiór IRIS\n")
        file.write("==========================\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification report:\n")
        file.write(report)

    print(f"\nWyniki zapisano w pliku: {results_path}")

    # Zadanie 6: kontrola wczytanej konfiguracji
    print("\n=== KONFIGURACJA ===")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"TEST_SIZE: {TEST_SIZE}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")


if __name__ == "__main__":
    main()
