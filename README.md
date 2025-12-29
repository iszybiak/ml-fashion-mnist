# Fashion-MNIST Classification Pipeline

## Opis projektu
Projekt przedstawia kompletny **pipeline uczenia maszynowego** do klasyfikacji obrazów ze zbioru **Fashion-MNIST** z wykorzystaniem konwolucyjnej sieci neuronowej (CNN).

Zakres projektu obejmuje:
- automatyczne pobieranie i przygotowanie danych,
- trening i walidację modelu CNN,
- ewaluację na zbiorze testowym,
- wizualizację metryk oraz błędnych predykcji.

Pipeline został zaprojektowany z myślą o **uruchamianiu wyłącznie na CPU**, bez konieczności użycia GPU.

---

## Struktura projektu
```
project/
├── data/
│   └── dataset.py        # Przygotowanie datasetów i DataLoaderów
├── model/
│   └── cnn.py            # Definicja architektury CNN
├── train.py              # Logika treningu i walidacji
├── evaluate.py           # Ewaluacja i wizualizacja wyników
├── main.py               # Uruchomienie pełnego pipeline’u
├── requirements.txt      # Lista zależności
└── README.md
```

Po uruchomieniu pipeline’u dane i model będą przechowywane lokalnie w folderach data/ i model/.

---

## Użyte technologie

- **Python 3.9+** – główny język implementacji.
- **PyTorch** – budowa modelu, trening, ewaluacja i optymalizacja.
- **torchvision** – pobieranie zbioru Fashion-MNIST oraz transformacje danych.
- **NumPy** – operacje numeryczne oraz kontrola reprodukowalności.
- **scikit-learn** – obliczanie metryk i macierzy pomyłek.
- **matplotlib** – wizualizacja wyników i błędnych predykcji.

---

## Wymagania

- Python 3.9–3.11  
- PyTorch 2.x  
- torchvision  
- numpy < 2 
- scikit-learn  
- matplotlib

---

## Instalacja 

Opcja 1: klonowanie repozytorium
```
git clone https://github.com/iszybiak/ml-fashion-mnist.git
cd ml-fashion-mnist
```
Opcja 2: pobranie archiwum
Pobierz projekt jako archiwum ZIP z GitHuba, rozpakuj go i przejdź do katalogu projektu:
```
cd ml-fashion-mnist
```

### Uruchomienie projektu
Windows
```
run.bat
```

## Przygotowanie danych
Zbiór Fashion-MNIST jest pobierany automatycznie przy pierwszym uruchomieniu.

Podział danych:
* Train: 80% danych treningowych
* Validation: 20% danych treningowych
* Test: pełny zbiór testowy

Walidacja:
Walidacja pozwala monitorować postęp modelu i unikać overfittingu – umożliwia wybór najlepszego modelu przed testem.
Podział 80/20 zapewnia wystarczającą liczbę przykładów do uczenia, przy zachowaniu reprezentatywnej próbki do walidacji.

Transformacje:
* Train: ToTensor(), normalizacja (0.5, 0.5), prosta augmentacja (RandomHorizontalFlip)
* Validation/Test: ToTensor(), normalizacja (0.5, 0.5)
* Augmentacja w treningu zwiększa różnorodność danych, poprawiając uogólnianie modelu.
* Brak augmentacji w walidacji/testach zapewnia spójność oceny modelu – porównujemy rzeczywiste dane z tymi, których model nie widział w treningu.
* Normalizacja pomaga szybciej trenować sieć, utrzymując wartości wejściowe w zakresie [-1, 1].

---

## Model

Prosty CNN:
* 2 warstwy konwolucyjne + ReLU + MaxPool
* 2 warstwy w pełni połączone
* spłaszczenie tensorów przy użyciu torch.flatten.
* Lekki, możliwy do trenowania na CPU w kilka minut

Decyzje projektowe:
* Architektura wybrana tak, aby balansować prostotę i skuteczność
* Fashion-MNIST to obrazki 28x28, więc nie potrzeba głębokich sieci ani transfer learningu.
* Dwie warstwy konwolucyjne pozwalają wyłapać proste cechy przestrzenne (krawędzie, tekstury).
* torch.flatten użyty do spłaszczenia tensorów

---

## Proces treningowy

* Funkcja straty: CrossEntropyLoss
* Optymalizator: Adam
* Batch size: 64 (wystarczająco duży do stabilnych gradientów na CPU)
* Epoki: 10 (równowaga między czasem a dokładnością)
* Walidacja po każdej epoce
* Early Stopping na podstawie accuracy walidacyjnej
* Zapis najlepszego modelu

Uzasadnienie:
* CrossEntropyLoss jest standardem dla klasyfikacji wieloklasowej.
* Adam zapewnia szybkie zbieganie i adaptacyjne dostosowanie learning rate.
* Trening po epoce z walidacją pozwala monitorować postęp uczenia i unikać przeuczenia.
* Zastosowano **Early Stopping** w celu uniknięcia overfittingu i skrócenia czasu treningu
* Zapis najlepszego modelu na podstawie walidacyjnej accuracy zapewnia stabilne wyniki testowe.
* Dodatkowe techniki, takie jak Dropout, Batch Normalization i augmentacja danych, były testowane w eksperymentach, ale w aktualnym pipeline pozostają zakomentowane.

---

## Ewaluacja
Podczas testu na zbiorze testowym raportowane są:
* accuracy i loss
* macierz pomyłek
* przykładowe błędnie sklasyfikowane obrazy.

Macierz pomyłek umożliwia identyfikację trudnych klas (np. T-shirt/top vs Shirt), a wizualizacja błędów pozwala na jakościową ocenę działania modelu.

---

## Wyniki

Model osiągnął accuracy na zbiorze testowym na poziomie **~91.4%**, 
co jest dobrym wynikiem dla prostej architektury CNN trenowanej wyłącznie na CPU.

Zaobserwowano:
* stabilny spadek funkcji straty,
* wzrost accuracy na zbiorach treningowym i walidacyjnym,
* brak istotnej luki pomiędzy walidacją a testem, co wskazuje na dobrą zdolność generalizacji.

Dodatkowo przeprowadzono mini-badania ograniczające overfitting, których szczegółowe wyniki (wykresy, porównania, wnioski) znajdują się w dołączonym raporcie PDF.

---

## Stabilność i reprodukowalnosć 
* Seed (42) dla numpy, random, torch
* Zapisywanie najlepszego modelu

Zapewnia to powtarzalność wyników na różnych maszynach.

---

## Dobre praktyki 
* Modularna struktura: dataset, model, trening, ewaluacja, pipeline
* Obsługa wyjątków w każdym module
* Zgodność z PEP8 i typami Python
* Brak `config.py` jest świadomym uproszczeniem wynikającym z charakteru projektu.

  
  




