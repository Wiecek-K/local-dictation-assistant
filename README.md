### **Dokumentacja Projektowa: Lokalny Asystent Dyktowania**

**Wersja:** 1.1
**Data:** 20.10.2025
**Autorzy Zmian:** [Twoje Imię/Inicjały], AI Architect

### Dziennik Zmian (Changelog)

- **Wersja 1.1 (20.10.2025):**
  - Dodano podsumowanie wyników fazy Proof of Concept (POC).
  - Zidentyfikowano krytyczne problemy z jakością transkrypcji w architekturze strumieniowej.
  - Potwierdzono, że model `small` na docelowym sprzęcie (GTX 1050 Ti) może osiągnąć zadowalającą jakość przy przetwarzaniu wsadowym (batch).
  - Zdiagnozowano problemy z ładowaniem modelu `medium` i powtórzeniami w generowanym tekście.
  - Zdefiniowano nowy priorytet: zaawansowany preprocessing audio jako klucz do maksymalizacji jakości modelu `small`.
  - Zaktualizowano plan działania, dodając etap rozwoju i testowania potoku preprocessingu.

---

### 1. Wprowadzenie i Cele Projektu

#### 1.1. Cel Główny

Celem projektu jest stworzenie wysokowydajnego narzędzia do transkrypcji mowy na tekst (Speech-to-Text), działającego w 100% lokalnie na systemie operacyjnym Linux (Pop!\_OS). Aplikacja ma służyć jako narzędzie wewnętrzne dla deweloperów, umożliwiając szybkie dyktowanie tekstu (np. kodu, komentarzy, wiadomości) bezpośrednio do dowolnego aktywnego pola tekstowego w systemie.

#### 1.2. Kluczowe Zasady Architektoniczne

- **Prywatność (Offline-First):** Żadne dane audio ani tekstowe nigdy nie opuszczają komputera użytkownika. Całe przetwarzanie odbywa się lokalnie.
- **Wydajność na Docelowym Sprzęcie:** Architektura musi być zoptymalizowana pod kątem działania na sprzęcie klasy Intel i5 z kartą graficzną Nvidia GTX 1050 Ti (4GB VRAM).
- **Głęboka Integracja Systemowa:** Aplikacja musi działać w tle i być aktywowana globalnym skrótem klawiszowym, a wynik jej pracy ma być natychmiast dostępny w aktywnym oknie.
- **Niska Odczuwalna Latencja:** Czas oczekiwania na tekst po zakończeniu dyktowania musi być zminimalizowany, nawet przy dyktowaniu dłuższych fragmentów.

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

### 3. Faza 1: Proof of Concept (POC) - Podsumowanie Wyników

#### 3.1. Cele POC (Status)

1.  **Weryfikacja Wydajności:** **Zakończone.** Potwierdzono, że docelowy sprzęt (GTX 1050 Ti) jest w stanie uruchamiać modele Whisper z akceleracją CUDA. Zidentyfikowano i rozwiązano liczne problemy konfiguracyjne środowiska.
2.  **Ocena Jakości:** **Zakończone.** Weryfikacja wykazała, że:
    - Architektura strumieniowa (streaming) w prostej implementacji **znacząco degraduje jakość** transkrypcji z powodu braku pełnego kontekstu.
    - Architektura wsadowa (batch, przetwarzanie całego pliku na raz) z modelem `small` **osiąga zadowalającą jakość bazową**.
    - Model `medium` oferuje wyższą jakość, ale jego działanie na docelowym sprzęcie jest niestabilne (problem z ładowaniem do VRAM, pętle powtórzeń).
3.  **Potwierdzenie Integracji:** **Zakończone.** Potwierdzono niezawodne działanie globalnego skrótu klawiszowego (`pynput`) i wklejania tekstu (`xdotool`).

#### 3.2. Kluczowe Wnioski z Fazy POC

- **Jakość > Niska Latencja:** Wstępne testy wykazały, że niska jakość transkrypcji jest większą barierą dla użyteczności niż kilkusekundowy czas oczekiwania. Dlatego priorytetem staje się maksymalizacja dokładności.
- **Preprocessing jest Kluczowy:** Analiza nagrań audio wykazała obecność szumów tła oraz problematycznych sybilantów i spółgłosek wybuchowych. Dostarczenie "czystszego" sygnału do modelu AI jest najbardziej obiecującą metodą na poprawę jakości bez zwiększania wymagań sprzętowych.
- **Model `small` jako Punkt Wyjścia:** Ze względu na stabilność i niskie zużycie zasobów, model `small` pozostaje naszym głównym celem. Celem jest osiągnięcie jakości zbliżonej do modelu `medium` poprzez zaawansowany preprocessing audio.

#### 3.3. Stos Technologiczny POC

- **Język:** Python 3.
- **Silnik STT:** `faster-whisper==0.10.0`
- **Akceleracja GPU:** `ctranslate2==3.24.0`
- **Nagrywanie Audio:** `sounddevice`
- **Przetwarzanie Audio:** `numpy`, `scipy`, `pydub`, `noisereduce`
- **Globalny Skrót Klawiszowy:** `pynput`
- **Integracja z Systemem:** `pyperclip`, `subprocess` z `xdotool`

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

Kluczowe jest zarządzanie oczekiwaniami co do wydajności. Poniższa tabela przedstawia szacowane parametry dla transkrypcji strumieniowej.

| Model (`faster-whisper`) | Zużycie VRAM (`int8`) | Szacowany RTF\* | Jakość (PL)  | Rekomendacja                                                         |
| :----------------------- | :-------------------- | :-------------- | :----------- | :------------------------------------------------------------------- |
| `base`                   | ~0.8 GB               | **0.3 - 0.4**   | Zadowalająca | Dobry wybór, jeśli szybkość jest absolutnym priorytetem.             |
| `small`                  | ~1.5 GB               | **0.5 - 0.7**   | **Dobra**    | **Rekomendowany.** Najlepszy kompromis między jakością a szybkością. |
| `medium`                 | ~3.0 GB               | ~1.2 - 1.5      | Bardzo dobra | Prawdopodobnie zbyt wolny (RTF > 1.0) i na granicy VRAM.             |

_\*RTF (Real-Time Factor): Czas Przetwarzania / Czas Trwania Audio. RTF = 0.5 oznacza, że 10 sekund audio jest przetwarzane w 5 sekund._

Dzięki architekturze strumieniowej, odczuwalny czas oczekiwania będzie zależał od długości ostatniego fragmentu audio (`chunk_seconds`), a nie od długości całego dyktowania, co jest kluczowe dla użyteczności aplikacji.

### 6. Potencjalne Kierunki Rozwoju (Roadmap)

- **Słownik niestandardowy:** Możliwość dodania przez użytkownika własnych słów (nazwy funkcji, terminologia techniczna) w celu poprawy dokładności.
- **Wsparcie dla Wayland:** Zbadanie i implementacja alternatyw dla `xdotool` i `pynput` dla nowszego serwera wyświetlania Wayland.
- **Zaawansowane formatowanie:** Automatyczne dodawanie znaków interpunkcyjnych i wielkich liter (funkcja dostępna w niektórych modelach).
- **Profilowanie wydajności:** Dodanie narzędzi do mierzenia realnego RTF i zużycia zasobów bezpośrednio w aplikacji konfiguracyjnej.
