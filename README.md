### **Dokumentacja Projektowa: Lokalny Asystent Dyktowania**

**Wersja:** 1.2
**Data:** 20.10.2025
**Autorzy Zmian:** [Ajmag], AI Architect

### Dziennik Zmian (Changelog)

- **Wersja 1.2 (20.10.2025):**
  - Zakończono fazę rozwoju i strojenia potoku przetwarzania wstępnego audio.
  - Zintegrowano finalny potok (Normalizacja, De-esser z wygładzaniem, Podbicie głośności, Odszumianie) z główną aplikacją.
  - Przeprowadzono testy porównawcze modeli `small` i `medium` na docelowym sprzęcie (GTX 1050 Ti).
  - Potwierdzono, że model `medium` z `compute_type='int8'` działa wydajnie (czas transkrypcji < 0.3s) i stabilnie (zużycie VRAM ~2.1 GB).
  - **Zmieniono rekomendowany model z `small` na `medium` w celu maksymalizacji jakości transkrypcji.**
- **Wersja 1.1 (20.10.2025):**
  - Dodano podsumowanie wyników fazy Proof of Concept (POC).
  - Zidentyfikowano krytyczne problemy z jakością transkrypcji w architekturze strumieniowej.
  - Zdefiniowano priorytet: zaawansowany preprocessing audio jako klucz do maksymalizacji jakości.

---

### 1. Wprowadzenie i Cele Projektu

#### 1.1. Cel Główny

Celem projektu jest stworzenie wysokowydajnego narzędzia do transkrypcji mowy na tekst (Speech-to-Text), działającego w 100% lokalnie na systemie operacyjnym Linux (Pop!\_OS). Aplikacja ma służyć jako narzędzie wewnętrzne dla deweloperów, umożliwiając szybkie dyktowanie tekstu (np. kodu, komentarzy, wiadomości) bezpośrednio do dowolnego aktywnego pola tekstowego w systemie.

#### 1.2. Kluczowe Zasady Architektoniczne

- **Prywatność (Offline-First):** Żadne dane audio ani tekstowe nigdy nie opuszczają komputera użytkownika.
- **Wydajność na Docelowym Sprzęcie:** Architektura musi być zoptymalizowana pod kątem działania na sprzęcie z kartą graficzną Nvidia GTX 1050 Ti (4GB VRAM).
- **Głęboka Integracja Systemowa:** Aplikacja musi działać w tle i być aktywowana globalnym skrótem klawiszowym.
- **Niska Odczuwalna Latencja:** Czas oczekiwania na tekst po zakończeniu dyktowania musi być zminimalizowany.

### 2. Architektura Wysokopoziomowa

System składa się z jednego głównego komponentu – **usługi działającej w tle (demona)** – która obsługuje cały proces od przechwycenia audio do wklejenia tekstu. Architektura opiera się na współbieżnym modelu **Producent-Konsument**, aby umożliwić transkrypcję strumieniową.

#### 2.1. Diagram Architektury Systemu

```
+--------------------------------------------------------------------------+
|                               System Operacyjny (Linux / X11)            |
|                                                                          |
|   +----------------------+         +---------------------------------+   |
|   | Użytkownik           |         | Aktywne Okno Aplikacji          |   |
|   | (np. Edytor Kodu)    |         | (np. VS Code, Przeglądarka)     |   |
|   +----------------------+         +---------------------------------+   |
|             |                                      ^                     |
|   1. Wciśnięcie i         +------------------+     | 5. Wklejenie tekstu |
|      przytrzymanie        | Globalny Listener|     |    (xdotool)       |
|      Skrótu Klawiszowego  | (pynput)         |     |                     |
|             |             +------------------+     |                     |
|             v                                      |                     |
|   +------------------------------------------------------------------+   |
|   |                        CORE SERVICE (Demon)                      |   |
|   |                                                                  |   |
|   |   2. Start Nagrywania i Transkrypcji                             |   |
|   |   +----------------------+      +-----------------------------+  |   |
|   |   | Wątek Nagrywający    |----->| Bezpieczna Wątkowo Kolejka  |  |   |
|   |   | (Producent)          |      | (queue)                     |  |   |
|   |   | - Przechwytuje audio |      +-----------------------------+  |   |
|   |   +----------------------+                  |                    |   |
|   |                                             v                    |   |
|   |   +----------------------------------------------------------+   |   |
|   |   | Wątek Transkrybujący (Konsument)                         |   |   |
|   |   | - Pobiera audio z kolejki                              |   |   |
|   |   | - Przetwarza w fragmentach (chunks) za pomocą Whisper    |   |   |
|   |   | - Wykorzystuje GPU (CUDA)                              |   |   |
|   |   | - Składa przetworzony tekst                            |   |   |
|   |   +----------------------------------------------------------+   |   |
|   |             |                                                    |   |
|   |             +--------------------------------------------------> |   |
|   |                                                                  |   |
|   |   4. Złożenie finalnego tekstu i skopiowanie do schowka (pyperclip)|   |
|   +------------------------------------------------------------------+   |
|                                                                          |
+--------------------------------------------------------------------------+
```

### 3. Faza 1: Rozwój i Strojenie Potoku Preprocessingu - Zakończone

Faza ta została zakończona sukcesem. Opracowano i wdrożono zaawansowany potok przetwarzania wstępnego audio, który znacząco poprawia jakość sygnału dostarczanego do modelu AI.

#### 3.1. Kluczowe Wnioski z Fazy R&D

- **Preprocessing jest kluczowy:** Testy A/B jednoznacznie wykazały, że "czysty" sygnał audio (po preprocessingu) pozwala modelom AI na osiągnięcie znacznie wyższej dokładności transkrypcji, eliminując błędy wynikające z artefaktów dźwiękowych (sybilanty, szum).
- **Preprocessing Umożliwia Użycie Modelu `medium`:** Najważniejszym odkryciem fazy jest fakt, że dzięki wysokiej jakości sygnału wejściowego, model `medium` staje się w pełni użyteczny na docelowym sprzęcie. Jego wyższa jakość językowa w połączeniu z czystym audio daje najlepsze rezultaty.
- **`int8` jest optymalnym typem obliczeń:** Testy potwierdziły, że kwantyzacja do `int8` oferuje najlepszy kompromis między zużyciem VRAM, szybkością a jakością na karcie GTX 1050 Ti. Próby użycia `float16` i `float32` zakończyły się błędami braku wsparcia lub pamięci.

#### 3.2. Finalny Potok Przetwarzania Wstępnego

1.  **Normalizacja Głośności:** Ujednolicenie poziomu sygnału.
2.  **Dynamiczny De-esser z Wygładzaniem:** Precyzyjne tłumienie ostrych sybilantów (`s`, `sz`, `cz`) bez wprowadzania artefaktów (trzasków).
3.  **Podbicie Głośności (Gain):** Zwiększenie ogólnej głośności w celu kompensacji i zapewnienia mocnego sygnału.
4.  **Redukcja Szumu:** Usunięcie stałego szumu tła.

#### 3.3. Stos Technologiczny

- **Język:** Python 3
- **Silnik STT:** `faster-whisper`
- **Akceleracja GPU:** `ctranslate2`
- **Przetwarzanie Audio:** `pydub`, `noisereduce`, `librosa`, `numpy`
- **Nagrywanie Audio:** `sounddevice`
- **Integracja z Systemem:** `pynput`, `pyperclip`, `subprocess` z `xdotool`

#### 3.4. Wyzwania Techniczne i Rozwiązania

Podczas fazy POC napotkano na szereg znaczących wyzwań związanych z konfiguracją środowiska deweloperskiego dla akceleracji GPU na platformie Linux. Dokumentacja tych problemów jest kluczowa dla przyszłego rozwoju i wdrożenia.

1.  **Problem: Niekompatybilność Typów Obliczeniowych (`compute_type`)**

    - **Objawy:** Aplikacja kończyła pracę z błędem `Requested float16 compute type, but the target device or backend do not support efficient float16 computation`, mimo że sprzęt (GTX 1050 Ti) wspiera tę technologię.
    - **Diagnoza:** Brak zainstalowanego pełnego **CUDA Toolkit** w systemie. Obecny był tylko sterownik Nvidia, co uniemożliwiało bibliotece `ctranslate2` poprawne wykrycie i wykorzystanie wszystkich możliwości karty graficznej.
    - **Rozwiązanie:** Instalacja pakietu `nvidia-cuda-toolkit` za pomocą menedżera pakietów `apt`.

2.  **Problem: Brak Biblioteki cuDNN**

    - **Objawy:** Aplikacja kończyła pracę błędem `SIGABRT` i komunikatem `Unable to load any of {libcudnn_ops.so...}`.
    - **Diagnoza:** Podstawowy pakiet `nvidia-cuda-toolkit` w dystrybucji Pop!\_OS nie zawiera biblioteki **cuDNN**, która jest kluczowa dla wydajności sieci neuronowych.
    - **Rozwiązanie:** Doinstalowanie dedykowanego pakietu `nvidia-cudnn` za pomocą `apt`.

3.  **Problem: Konflikt Wersji cuDNN (Wbudowana vs. Systemowa)**

    - **Objawy:** Mimo obecności systemowej biblioteki `libcudnn.so.8`, aplikacja nadal zgłaszała błąd o braku `libcudnn.so.9`.
    - **Diagnoza:** Analiza za pomocą `ldd` wykazała, że domyślnie instalowana wersja `ctranslate2` (4.x) jest spakowana z **własną, wbudowaną wersją cuDNN 9**, która okazała się niekompatybilna z architekturą GPU (Pascal) i systemowym CUDA Toolkit (v11.5).
    - **Rozwiązanie:** Zastosowano strategię "cofnięcia w czasie" (version rollback). Ręcznie zainstalowano starsze, ale wzajemnie kompatybilne wersje bibliotek: `ctranslate2==3.24.0` i `faster-whisper==0.10.0`.

4.  **Problem: Brak Zależności Systemowych dla `PyAV`**
    - **Objawy:** Błąd kompilacji podczas instalacji biblioteki `av`, wymaganej przez `faster-whisper`, z komunikatem `Package libavformat was not found`.
    - **Diagnoza:** Brak zainstalowanych w systemie bibliotek deweloperskich **FFmpeg**.
    - **Rozwiązanie:** Instalacja pakietu `ffmpeg` za pomocą `apt`.

**Wniosek z problemów konfiguracyjnych:** Środowisko AI na platformie Linux z akceleracją Nvidia jest wysoce wrażliwe na wersje sterowników, CUDA Toolkit, cuDNN oraz sposób kompilacji bibliotek Pythona. Kluczowe jest zapewnienie spójności między tymi wszystkimi komponentami.

### 4. Faza 2: Aplikacja Docelowa

Faza rozwoju zostaje podzielona na dwa główne etapy:

1.  **Etap 1: Rozwój i Strojenie Potoku Przetwarzania Wstępnego Audio (Obecny Priorytet)**
2.  **Etap 2: Budowa Aplikacji Docelowej**

#### 4.1. Etap 1: Rozwój Potoku Przetwarzania Wstępnego (Preprocessing Pipeline)

**Cel:** Stworzenie i przetestowanie zestawu filtrów audio, które znacząco poprawią jakość nagrań dostarczanych do modelu Whisper.

**Zadania:**

1.  **Stworzenie Środowiska Testowego:**
    - Przygotowanie dedykowanego skryptu (`test_preprocessing.py`) do szybkiego testowania filtrów.
    - Nagranie zestawu referencyjnych plików audio (`.wav`) zawierających zidentyfikowane problemy (sybilanty, spółgłoski wybuchowe, szum).
2.  **Implementacja Potoku Przetwarzania:**
    - **Normalizacja Głośności:** Zapewnienie optymalnego i spójnego poziomu sygnału.
    - **De-ploser (Filtr Górnoprzepustowy):** Redukcja niskoczęstotliwościowych "uderzeń" od spółgłosek wybuchowych.
    - **De-esser (Filtr Sybilantów):** Kontrola nad ostrymi, syczącymi głoskami.
    - **Odszumianie (Noise Reduction):** Redukcja stałego szumu tła.
3.  **Testowanie i Strojenie:**
    - Wykorzystanie metodologii testowania (analiza wizualna spektrogramów, testy odsłuchowe A/B, porównanie wyników transkrypcji) do znalezienia optymalnych parametrów dla każdego filtra.
    - Celem jest osiągnięcie balansu między redukcją artefaktów a zachowaniem naturalnego brzmienia głosu.
4.  **Integracja z Główną Aplikacją:**
    - Przeniesienie dopracowanego potoku preprocessingu do skryptu `main_simple.py`.
    - Finalna weryfikacja jakości transkrypcji na modelu `small` z przetworzonym audio.

**Kryteria Sukcesu Etapu 1:**

- Transkrypcja z modelu `small` na przetworzonym audio jest obiektywnie (mniej błędów) i subiektywnie (bardziej naturalna) lepsza niż z modelu `medium` na surowym audio.
- Użytkownik nie musi "mówić wyraźnie do mikrofonu", aby uzyskać dobrą jakość.

#### 4.2. Zakres Funkcjonalny i Ulepszenia

- **Demon Systemowy:** Główna logika aplikacji jest spakowana jako usługa `systemd`, która startuje automatycznie wraz z systemem.
- **Aplikacja Konfiguracyjna (GUI):** Stworzenie prostego interfejsu graficznego (np. w PyQt lub GTK), który pozwala na:
  - Graficzny wybór i rejestrację skrótu klawiszowego.
  - Wybór modelu z listy (z opcją pobrania nowych).
  - Wybór języka i urządzenia (CPU/GPU).
  - Wyświetlanie statusu usługi (działa / nie działa).
  - Uruchamianie, zatrzymywanie i restartowanie usługi.
- **Instalator:** Stworzenie pakietu `.deb`, który automatyzuje proces instalacji:
  - Instaluje zależności systemowe (`xdotool`, `python3-pip`).
  - Instaluje biblioteki Pythona.
  - Konfiguruje i aktywuje usługę `systemd`.
  - Dodaje skrót do aplikacji konfiguracyjnej w menu systemowym.
- **Zadanie 1: Powrót do Architektury Strumieniowej:** Implementacja `main_streaming.py` z wykorzystaniem dopracowanego potoku preprocessingu na każdym fragmencie (chunk).
- **Zadanie 2: Demon Systemowy:** Spakowanie aplikacji jako usługa `systemd`.
- **Zadanie 3: Aplikacja Konfiguracyjna (GUI):** Stworzenie interfejsu do zarządzania ustawieniami.
- **Zadanie 4: Instalator:** Przygotowanie pakietu `.deb`.

---

#### 4.3. Architektura Aplikacji Docelowej

```
+---------------------------------------------------------------------------------+
|                                  Użytkownik                                     |
+---------------------------------------------------------------------------------+
       |                                      ^
       | Interakcja (zmiana ustawień)         | Informacja o statusie
       v                                      |
+--------------------------------------+     +-------------------------------------+
| Aplikacja Konfiguracyjna (GUI)       |     | Core Service (Demon systemd)        |
| - UI (PyQt/GTK)                      |     | - Uruchamiany przy starcie systemu  |
| - Wybór skrótu, modelu, języka       |     | - Zawiera całą logikę z POC         |
| - Zarządzanie usługą (start/stop)    |     |   (Listener, Wątki, Transkrypcja)   |
+--------------------------------------+     +-------------------------------------+
       |                                      ^
       | 1. Zapisuje zmiany                   | 2. Odczytuje konfigurację
       |    do pliku konfiguracyjnego         |    przy starcie lub na sygnał
       v                                      |
+---------------------------------------------------------------------------------+
|                                Plik config.ini                                  |
+---------------------------------------------------------------------------------+
```

### 5. Analiza Wydajności na Docelowym Sprzęcie (GTX 1050 Ti)

Testy przeprowadzone po fazie R&D dostarczyły nowych, precyzyjnych danych na temat wydajności.

| Model (`faster-whisper`) | Zużycie VRAM (`int8`) | Czas Transkrypcji (plik 30s) | Jakość (PL)      | Rekomendacja                                                       |
| :----------------------- | :-------------------- | :--------------------------- | :--------------- | :----------------------------------------------------------------- |
| `small`                  | ~1.4 GB               | **~0.2s**                    | Dobra            | Dobry wybór dla systemów z bardzo ograniczonym VRAM (< 4GB).       |
| `medium`                 | ~2.1 GB               | **~0.2s**                    | **Bardzo Dobra** | **REKOMENDOWANY.** Najlepszy kompromis między jakością a zasobami. |

**Wnioski:**

- Czas samej transkrypcji dla obu modeli jest pomijalnie niski i nieodczuwalny dla użytkownika.
- Głównym kosztem jest zużycie VRAM i jednorazowy czas ładowania modelu przy starcie aplikacji.
- Model `medium` zużywa ~52% VRAM karty GTX 1050 Ti, co jest w pełni akceptowalnym i bezpiecznym poziomem.

### 6. Potencjalne Kierunki Rozwoju (Roadmap)

- **Słownik niestandardowy:** Możliwość dodania przez użytkownika własnych słów (nazwy funkcji, terminologia techniczna) w celu poprawy dokładności.
- **Wsparcie dla Wayland:** Zbadanie i implementacja alternatyw dla `xdotool` i `pynput` dla nowszego serwera wyświetlania Wayland.
- **Zaawansowane formatowanie:** Automatyczne dodawanie znaków interpunkcyjnych i wielkich liter (funkcja dostępna w niektórych modelach).
- **Profilowanie wydajności:** Dodanie narzędzi do mierzenia realnego RTF i zużycia zasobów bezpośrednio w aplikacji konfiguracyjnej.
- **Inteligentna edycja tekstu (Post-processing):** Możliwość wykorzystania małego, lokalnego modelu językowego (LLM) do automatycznego usuwania pomyłek, powtórzeń i wahań z surowej transkrypcji.
