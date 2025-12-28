# Fashion-MNIST Classification Pipeline

## Opis projektu
Projekt jest kompletnym pipeline'em uczenia maszynowego do klasyfikacji obrazów z wykorzystaniem zbioru **Fashion-MNIST**.  
Pipeline obejmuje:
* Pobranie i przygotowanie danych
* Trening prostego **CNN** 
* Walidację i ewaluację modelu
* Wyświetlanie metryk i wizualizację wyników

Projekt został zaprojektowany tak, aby działał **na CPU**, bez wymogu GPU, zgodnie z wymaganiami. 

---

## Struktura projektu

project/
├── data/
│ └── dataset.py # Przygotowanie DataLoaderów i transformacji
├── model/
│ └── cnn.py # Definicja CNN
├── train.py # Trening modelu
├── evaluate.py # Ewaluacja modelu, wizualizacja
├── main.py # Uruchomienie treningu i ewaluacji
├── requirements.txt # Lista wymaganych bibliotek
└── README.md

Po uruchomieniu pipeline’u dane i model będą przechowywane lokalnie w folderach data/ i model/.

## Wymagania

- Python 3.9+
- PyTorch 2.x
- torchvision 0.x
- numpy 1.21+
- scikit-learn
- matplotlib

## Użyte technologie

- **Python 3.9+**
  Główny język programowania użyty do implementacji pipeline’u ML.

- **PyTorch**
  Framework deep learningowy użyty do:
  - budowy modelu CNN
  - treningu i ewaluacji
  - obsługi autograd i optymalizacji

- **torchvision**
  Wykorzystany do:
  - pobrania zbioru Fashion-MNIST
  - transformacji danych (normalizacja, augmentacja)

- **NumPy**
  Operacje numeryczne oraz wsparcie dla reproducibility (seed).

- **scikit-learn**
  Obliczanie metryk oraz wizualizacja macierzy pomyłek.

- **matplotlib**
  Wizualizacja wyników:
  - confusion matrix
  - błędnie sklasyfikowane próbki


### Instalacja 

```
pip install -r requirements.txt
```

## Instrukcje uruchomienia
Uruchom pipeline treningowy i ewaluację:
```
python run_pipeline.py
```
Trening zapisuje najlepszy model do:
```
./model/best_model.pth
```

W projekcie ustawiony jest seed (`42`) dla:
- random
- numpy
- torch

Ewaluacja wyświetla:
* Accuracy i loss
* Macierz pomyłek
* Wybrane błędnie sklasyfikowane próbki

## Przygotowanie danych
Zbiór Fashion-MNIST jest pobierany automatycznie.

Podział danych:
* Train: 80% danych treningowych
* Validation: 20% danych treningowych
* Test: pełny zbiór testowy

Transformacje:
* ToTensor() i normalizacja (0.5, 0.5) dla wszystkich zbiorów
* Prosta augmentacja (RandomHorizontalFlip) tylko dla zbioru treningowego

Walidacja:
Walidacja pozwala monitorować postęp modelu i unikać overfittingu – umożliwia wybór najlepszego modelu przed testem.
Podział 80/20 zapewnia wystarczającą liczbę przykładów do uczenia, przy zachowaniu reprezentatywnej próbki do walidacji.

Transformacje:
* Train: ToTensor(), normalizacja (0.5, 0.5), prosta augmentacja (RandomHorizontalFlip)
* Validation/Test: ToTensor(), normalizacja (0.5, 0.5)
* Augmentacja w treningu zwiększa różnorodność danych, poprawiając uogólnianie modelu.
* Brak augmentacji w walidacji/testach zapewnia spójność oceny modelu – porównujemy rzeczywiste dane z tymi, których model nie widział w treningu.
* Normalizacja pomaga szybciej trenować sieć, utrzymując wartości wejściowe w zakresie [-1, 1].

## Model

Prosty CNN:
* 2 warstwy konwolucyjne + ReLU + MaxPool
* 2 warstwy w pełni połączone
* Lekki, możliwy do trenowania na CPU w kilka minut

Decyzje projektowe:
* Architektura wybrana tak, aby balansować prostotę i skuteczność
* Fashion-MNIST to obrazki 28x28, więc nie potrzeba głębokich sieci ani transfer learningu.
* Dwie warstwy konwolucyjne pozwalają wyłapać proste cechy przestrzenne (krawędzie, tekstury).
* torch.flatten użyty do spłaszczenia tensorów

## Proces treningowy

* Funkcja straty: CrossEntropyLoss
* Optymalizator: Adam
* Batch size: 64 (wystarczająco duży do stabilnych gradientów na CPU)
* Epoki: 10 (równowaga między czasem a dokładnością)

Uzasadnienie:
* CrossEntropyLoss jest standardem dla klasyfikacji wieloklasowej.
* Adam zapewnia szybkie zbieganie i adaptacyjne dostosowanie learning rate.
* Trening po epoce z walidacją pozwala monitorować postęp uczenia i unikać przeuczenia.
* Aby uniknąć overfittingu i skrócić czas treningu, zastosowano **Early Stopping**
* Zapis najlepszego modelu na podstawie walidacyjnej accuracy zapewnia stabilne wyniki testowe.
* Dodatkowe techniki, takie jak Dropout, Batch Normalization i augmentacja danych, były testowane w eksperymentach, ale w aktualnym pipeline pozostają zakomentowane.

## Ewaluacja
* Test na pełnym zbiorze testowym, raportowanie accuracy i loss
* Wizualizacja:
  * Macierz pomyłek (Confusion Matrix)
  * Wybrane błędnie sklasyfikowane próbki

Uzasadnienie:
* Macierz pomyłek pozwala zidentyfikować trudne klasy, np. T-shirt/top vs Shirt.
* Wyświetlenie błędnie sklasyfikowanych obrazów pozwala łatwo zweryfikować praktyczne działanie modelu.

# Krótka interpretacja uzykanych wyników

Model osiągnął accuracy na zbiorze testowym na poziomie **~91.4%**, 
co jest dobrym wynikiem dla prostej architektury CNN trenowanej wyłącznie na CPU.

Podczas treningu obserwowany był stabilny spadek funkcji straty oraz wzrost accuracy
zarówno na zbiorze treningowym, jak i walidacyjnym.
Brak znaczącej różnicy pomiędzy wynikami walidacyjnymi i testowymi
wskazuje na dobrą zdolność generalizacji modelu oraz brak silnego overfittingu.

W ramach projektu przeprowadzono również **mini badania mające na celu ograniczenie overfittingu**
oraz poprawę zdolności generalizacji modelu.
Szczegółowy opis przeprowadzonych eksperymentów, wraz z wykresami i wnioskami,
został przedstawiony w osobnym raporcie w formacie **PDF**, dołączonym do repozytorium.


## Stabilność i reproducibility
* Seed (42) dla numpy, random, torch
* Zapisywanie modelu

Uzasadnienie:
* Seed zapewnia, że wyniki są powtarzalne na różnych komputerach.


## Dobre praktyki kodu

* Modularna struktura: dataset, model, trening, ewaluacja, pipeline
* Obsługa wyjątków w każdym module
* Zgodność z PEP8 i typami Python

Uzasadnienie:
* Ułatwia utrzymanie kodu i rozwój projektu
* Pozwala na szybkie diagnozowanie błędów i śledzenie metryk
* Zapewnia czytelność
* Brak `config.py` jest świadomym uproszczeniem wynikającym z charakteru projektu.
  




