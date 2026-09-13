# Date Normalizer

Projekt rozwiązuje problem normalizacji nieregularnych ciągów tekstowych reprezentujących daty. Jest to zadanie translacji maszynowej klasy Sequence-to-Sequence (Seq2Seq), w którym model odczytuje datę wejściową o zmiennej długości i generuje poprawną datę w formacie ISO 8601 (YYYY-MM-DD).

## ⚙️ Architektura Modelu
Model został zaimplementowany od podstaw w bibliotece PyTorch jako pełna architektura **Transformer Encoder-Decoder**, bez użycia gotowych pipeline'ów.

* **Encoder (Czytacz):** 1 warstwa z mechanizmem Self-Attention, siecią Feed-Forward (FFN) oraz schematem Add & Norm.
* **Decoder (Pisarz):** Autoregresyjna generacja znaków (znak po znaku)[cite: 1]. Zawiera Masked Self-Attention (z maską trójkątną), Cross-Attention (komunikacja z koderem) oraz sieć FFN.
* **Tokenizacja:** Char-level (na poziomie znaków)[cite: 1]. Słownik (vocab) ogranicza się do 39 unikalnych tokenów (w tym tagi specjalne: `<PAD>`, `<START>`, `<END>`).
* **Hiperparametry:** `d_model = 64`, `num_heads = 4`, `batch_size = 32`.

## 📊 Zbiór Danych
Zaimplementowano autorski generator w języku Python, który stworzył syntetyczny zbiór **20 000 par** dat (wejście, wyjście)[cite: 1]. Zbiór podzielono na 18 000 przykładów treningowych (90%) i 2000 walidacyjnych (10%) pakowanych przy użyciu klasy DataLoader.

Obsługiwane formaty wejściowe to m.in.:
* `15 marca 2026` (słowny zapis miesiąca)
* `05.03.2026` (format DD.MM.YYYY)
* `5/3/2026` (format D/M/YYYY)
* Format mieszany (np. `05.3.2026`, `5/03/2026`)

## 📈 Trening i Ewaluacja
Model trenowano przez 10 epok przy użyciu optymalizatora Adam (`lr = 0.001`) oraz funkcji straty CrossEntropyLoss (z ignorowaniem tokenu `<PAD>`) z zastosowaniem techniki Teacher Forcing.

Do oceny jakości zastosowano metrykę **Exact Match Accuracy**[cite: 1]. Na 100 niewidzianych podczas treningu, losowych przykładach, model uzyskał skuteczność **100.00%**[cite: 1]. Training Loss i Validation Loss równolegle zbiegły do wartości bliskiej zeru około 7. epoki, co świadczy o stabilnej zbieżności i braku przeuczenia.

## Przykłady działania (Inference)

| Wejście | Wynik (ISO 8601) |
| :--- | :--- |
| `15 marca 2026` | `2026-03-15` |
| `04.12.2020` | `2020-12-04` |
| `28 lutego 2005` | `2005-02-28` |
| `1/01/2030` | `2030-01-01` |
| `5.3.2025` | `2025-03-05` |

## Technologie i Biblioteki
* **PyTorch** (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`) – budowa modelu, tokenizacja, mechanizm uwagi i trening.
* **Matplotlib** (`matplotlib.pyplot`) – generowanie wykresów przebiegu uczenia.
* **Seaborn** (`seaborn`) – tworzenie map ciepła (heatmap) wizualizujących wagi z mechanizmu Cross-Attention.
* **Python Built-ins** (`random`) – generowanie w locie losowych przypadków testowych datasetu.
