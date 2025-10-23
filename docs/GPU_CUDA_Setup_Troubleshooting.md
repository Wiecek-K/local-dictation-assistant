# Dokumentacja Problemów: Konfiguracja GPU, CUDA i Bibliotek Python

Data: 2025-10-23
System: Pop!\_OS 22.04 LTS
GPU: NVIDIA GeForce GTX 1050 Ti Mobile

## Podsumowanie

Podczas konfiguracji środowiska dla projektu `local-dictation-assistant` napotkano na szereg krytycznych problemów uniemożliwiających bibliotece `faster-whisper` wykorzystanie akceleracji GPU. Problemy te wynikały z fundamentalnych niespójności między sterownikiem NVIDIA, zainstalowanymi komponentami CUDA a wersjami bibliotek Python.

Poniżej znajduje się opis problemów i ich ostateczne rozwiązania, które doprowadziły do stabilnej konfiguracji.

---

### Problem 1: Błędy `compute_type` (`float16`, `int8`)

- **Objawy:** Aplikacja kończyła pracę z błędem `Requested float16 compute type, but the target device or backend do not support efficient float16 computation`, mimo że sprzęt (architektura Pascal) technicznie wspiera te operacje.
- **Diagnoza:** W systemie zainstalowany był jedynie **sterownik NVIDIA**, ale brakowało pełnego **CUDA Toolkit**. Sterownik informuje system o _możliwościach_ karty, ale to Toolkit dostarcza kompilatory (`nvcc`) i biblioteki deweloperskie niezbędne dla aplikacji AI do faktycznego wykorzystania tych możliwości.
- **Rozwiązanie:**
  1.  Instalacja pełnego środowiska deweloperskiego CUDA za pomocą menedżera pakietów:
      ```bash
      sudo apt install nvidia-cuda-toolkit
      ```

### Problem 2: Brak Biblioteki cuDNN

- **Objawy:** Aplikacja kończyła pracę błędem `SIGABRT` i komunikatem `Unable to load any of {libcudnn_ops.so...}`.
- **Diagnoza:** Podstawowy pakiet `nvidia-cuda-toolkit` w repozytoriach Ubuntu/Pop!\_OS nie zawiera biblioteki **cuDNN**, która jest kluczowa dla wydajności operacji na sieciach konwolucyjnych (CNN), będących częścią modelu Whisper.
- **Rozwiązanie:**
  1.  Doinstalowanie dedykowanego pakietu `nvidia-cudnn`:
      ```bash
      sudo apt install nvidia-cudnn
      ```

### Problem 3: Konflikt Wersji Bibliotek Python (`pip freeze` vs. "Magiczny Przepis")

- **Objawy:** Po odtworzeniu środowiska wirtualnego z pliku `requirements.txt` wygenerowanego przez `pip freeze`, pojawiała się lawina błędów o konfliktach zależności i niekompatybilnych wersjach.
- **Diagnoza:**
  1.  **Konflikt Wersji `ctranslate2`:** Domyślnie instalowana wersja `ctranslate2` (4.x) była spakowana z **wbudowaną biblioteką cuDNN 9**, która jest niekompatybilna z naszym systemowym CUDA Toolkit (v11.5) i GPU (architektura Pascal).
  2.  **Pułapka `pip freeze`:** `pip freeze` zapisuje "płaską" listę wszystkich pakietów, ale nie zachowuje informacji o **kolejności instalacji** ani o **specjalnych flagach** (jak `--no-dependencies`), które były kluczowe do obejścia konfliktów.
- **Rozwiązanie ("Magiczny Przepis"):**

  1.  Stworzenie **minimalnego pliku `requirements.txt`** zawierającego tylko bezpośrednie zależności.
  2.  Zastosowanie **ścisłej kolejności instalacji**, która ręcznie rozwiązuje konflikty:

      ```bash
      # 1. Zainstaluj starszą, kompatybilną wersję ctranslate2
      pip install "ctranslate2==3.24.0"

      # 2. Zainstaluj starszą wersję faster-whisper, IGNORUJĄC jej zależności
      pip install --no-dependencies "faster-whisper==0.10.0"

      # 3. Zainstaluj wszystkie pozostałe zależności, które teraz dopasują się do istniejących pakietów
      pip install -r requirements.txt
      ```

### Wnioski i Rekomendacje

- Środowisko AI na Linuksie jest ekosystemem, w którym kluczowa jest **spójność wersji** między sterownikiem, CUDA Toolkit, cuDNN i bibliotekami Pythona.
- **Nigdy nie ufaj ślepo `pip freeze`** w projektach z tak złożonymi zależnościami. Zamiast tego, utrzymuj **minimalny plik `requirements.txt`** i dokumentuj proces instalacji w osobnym pliku (np. `INSTALL.md`).
- W przypadku problemów, kluczowe komendy diagnostyczne to `nvidia-smi`, `nvcc --version` oraz `ldd` do sprawdzania powiązań bibliotek.
