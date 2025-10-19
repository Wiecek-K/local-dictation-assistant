### **Dokumentacja Projektowa: Lokalny Asystent Dyktowania**
**Wersja:** 1.0
**Data:** 19.10.2025

### 1. Wprowadzenie i Cele Projektu

#### 1.1. Cel Główny
Celem projektu jest stworzenie wysokowydajnego narzędzia do transkrypcji mowy na tekst (Speech-to-Text), działającego w 100% lokalnie na systemie operacyjnym Linux (Pop!_OS). Aplikacja ma służyć jako narzędzie wewnętrzne dla deweloperów, umożliwiając szybkie dyktowanie tekstu (np. kodu, komentarzy, wiadomości) bezpośrednio do dowolnego aktywnego pola tekstowego w systemie.

#### 1.2. Kluczowe Zasady Architektoniczne
*   **Prywatność (Offline-First):** Żadne dane audio ani tekstowe nigdy nie opuszczają komputera użytkownika. Całe przetwarzanie odbywa się lokalnie.
*   **Wydajność na Docelowym Sprzęcie:** Architektura musi być zoptymalizowana pod kątem działania na sprzęcie klasy Intel i5 z kartą graficzną Nvidia GTX 1050 Ti (4GB VRAM).
*   **Głęboka Integracja Systemowa:** Aplikacja musi działać w tle i być aktywowana globalnym skrótem klawiszowym, a wynik jej pracy ma być natychmiast dostępny w aktywnym oknie.
*   **Niska Odczuwalna Latencja:** Czas oczekiwania na tekst po zakończeniu dyktowania musi być zminimalizowany, nawet przy dyktowaniu dłuższych fragmentów.

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

### 3. Faza 1: Proof of Concept (POC)

#### 3.1. Cele POC
1.  **Weryfikacja Wydajności:** Sprawdzenie, czy transkrypcja strumieniowa na docelowym sprzęcie (GTX 1050 Ti) zapewnia akceptowalną, niską odczuwalną latencję.
2.  **Ocena Jakości:** Ustalenie, czy jakość transkrypcji modeli `base` i `small` (z kwantyzacją `int8`) jest wystarczająca dla języka polskiego.
3.  **Potwierdzenie Integracji:** Udowodnienie, że mechanizmy globalnego skrótu klawiszowego i wklejania tekstu działają niezawodnie w środowisku Pop!_OS.

#### 3.2. Zakres Funkcjonalny POC
*   Aplikacja uruchamiana jako pojedynczy skrypt w Pythonie.
*   Konfiguracja (model, skrót, język) odbywa się poprzez edycję pliku `config.ini`.
*   Implementacja transkrypcji strumieniowej z użyciem wątków i kolejki.
*   Wsparcie dla VAD (Voice Activity Detection) w celu usuwania ciszy.
*   Wynikowy tekst jest kopiowany do schowka i automatycznie wklejany.
*   Brak interfejsu graficznego.

#### 3.3. Stos Technologiczny POC
*   **Język:** Python 3.
*   **Silnik STT:** `faster-whisper` (dla optymalizacji CUDA i VAD).
*   **Akceleracja GPU:** `CTranslate2` (backend dla `faster-whisper`).
*   **Nagrywanie Audio:** `sounddevice`.
*   **Globalny Skrót Klawiszowy:** `pynput`.
*   **Integracja z Systemem:** `pyperclip` (schowek), `subprocess` z `xdotool` (wklejanie).

#### 3.4. Kryteria Sukcesu POC
*   Odczuwalny czas oczekiwania po zakończeniu 30-sekundowego dyktowania jest krótszy niż 5 sekund.
*   Aplikacja działa stabilnie i nie powoduje zauważalnego spowolnienia systemu podczas dyktowania.
*   Dokładność transkrypcji modelu `small` jest oceniana jako "dobra" lub "bardzo dobra" przez wewnętrznych testerów.

### 4. Faza 2: Aplikacja Docelowa

#### 4.1. Cel
Przekształcenie działającego POC w stabilne, łatwe w instalacji i zarządzaniu narzędzie, które jest integralną częścią systemu operacyjnego.

#### 4.2. Zakres Funkcjonalny i Ulepszenia
*   **Demon Systemowy:** Główna logika aplikacji jest spakowana jako usługa `systemd`, która startuje automatycznie wraz z systemem.
*   **Aplikacja Konfiguracyjna (GUI):** Stworzenie prostego interfejsu graficznego (np. w PyQt lub GTK), który pozwala na:
    *   Graficzny wybór i rejestrację skrótu klawiszowego.
    *   Wybór modelu z listy (z opcją pobrania nowych).
    *   Wybór języka i urządzenia (CPU/GPU).
    *   Wyświetlanie statusu usługi (działa / nie działa).
    *   Uruchamianie, zatrzymywanie i restartowanie usługi.
*   **Instalator:** Stworzenie pakietu `.deb`, który automatyzuje proces instalacji:
    *   Instaluje zależności systemowe (`xdotool`, `python3-pip`).
    *   Instaluje biblioteki Pythona.
    *   Konfiguruje i aktywuje usługę `systemd`.
    *   Dodaje skrót do aplikacji konfiguracyjnej w menu systemowym.

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

| Model (`faster-whisper`) | Zużycie VRAM (`int8`) | Szacowany RTF* | Jakość (PL) | Rekomendacja |
| :--- | :--- | :--- | :--- | :--- |
| `base` | ~0.8 GB | **0.3 - 0.4** | Zadowalająca | Dobry wybór, jeśli szybkość jest absolutnym priorytetem. |
| `small` | ~1.5 GB | **0.5 - 0.7** | **Dobra** | **Rekomendowany.** Najlepszy kompromis między jakością a szybkością. |
| `medium` | ~3.0 GB | ~1.2 - 1.5 | Bardzo dobra | Prawdopodobnie zbyt wolny (RTF > 1.0) i na granicy VRAM. |

*\*RTF (Real-Time Factor): Czas Przetwarzania / Czas Trwania Audio. RTF = 0.5 oznacza, że 10 sekund audio jest przetwarzane w 5 sekund.*

Dzięki architekturze strumieniowej, odczuwalny czas oczekiwania będzie zależał od długości ostatniego fragmentu audio (`chunk_seconds`), a nie od długości całego dyktowania, co jest kluczowe dla użyteczności aplikacji.

### 6. Potencjalne Kierunki Rozwoju (Roadmap)
*   **Słownik niestandardowy:** Możliwość dodania przez użytkownika własnych słów (nazwy funkcji, terminologia techniczna) w celu poprawy dokładności.
*   **Wsparcie dla Wayland:** Zbadanie i implementacja alternatyw dla `xdotool` i `pynput` dla nowszego serwera wyświetlania Wayland.
*   **Zaawansowane formatowanie:** Automatyczne dodawanie znaków interpunkcyjnych i wielkich liter (funkcja dostępna w niektórych modelach).
*   **Profilowanie wydajności:** Dodanie narzędzi do mierzenia realnego RTF i zużycia zasobów bezpośrednio w aplikacji konfiguracyjnej.
