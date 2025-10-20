# /testing_suite/run_comparison_test.py
# Wersja 2: Poprawione ścieżki, zakładając, że skrypt jest w podkatalogu,
# a plik config.ini znajduje się w głównym katalogu projektu.

import sys
import time
import configparser
import librosa
from faster_whisper import WhisperModel
import os

# --- Konfiguracja Ścieżek ---
# Skrypt znajduje się w 'testing_suite', więc musimy odwołać się do katalogu nadrzędnego
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Główny katalog projektu jest teraz dwa poziomy wyżej
ROOT_DIR = os.path.dirname(PARENT_DIR)

CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')
RAW_AUDIO_PATH = os.path.join(PARENT_DIR, 'sibilants_test.wav')
PROCESSED_AUDIO_PATH = os.path.join(PARENT_DIR, 'test_outputs', 'sibilants_test_processed.wav')

def load_configuration(override_model_path=None):
    """Wczytuje konfigurację i opcjonalnie nadpisuje ścieżkę modelu."""
    config = configparser.ConfigParser()
    try:
        config.read(CONFIG_PATH)
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='small'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
            'language': config.get('settings', 'language', fallback='pl'),
        }
        if override_model_path:
            print(f"INFO: Nadpisywanie modelu z config.ini na: '{override_model_path}'")
            settings['model_path'] = override_model_path
        return settings
    except Exception as e:
        print(f"Błąd wczytywania {CONFIG_PATH}: {e}")
        sys.exit(1)

def load_model(settings):
    """Wczytuje i zwraca model Whisper na podstawie ustawień."""
    model_path = settings['model_path']
    print(f"\n--- Ładowanie Modelu '{model_path}' ---")
    start_time = time.time()
    try:
        model = WhisperModel(model_path, device=settings['device'], compute_type=settings['compute_type'])
        print(f"✅ Model załadowany w {time.time() - start_time:.2f}s.")
        return model
    except Exception as e:
        print(f"❌ BŁĄD KRYTYCZNY: Nie udało się załadować modelu: {e}")
        return None

def transcribe_audio(model, file_path, language):
    """Transkrybuje pojedynczy plik audio i zwraca tekst."""
    if not os.path.exists(file_path):
        print(f"❌ BŁĄD: Plik audio nie istnieje: {file_path}")
        if "processed" in file_path:
            print("   -> Upewnij się, że najpierw uruchomiłeś 'test_preprocessing.py', aby wygenerować ten plik.")
        return "[BŁĄD: BRAK PLIKU]"
        
    print(f"   -> Transkrypcja pliku: {os.path.basename(file_path)}...")
    try:
        audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
        segments, _ = model.transcribe(audio_data, language=language, beam_size=5)
        return "".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"   -> ❌ Błąd podczas transkrypcji: {e}")
        return f"[BŁĄD TRANSKRYPCJI: {e}]"

def main():
    print("--- Uruchamianie Pełnego Testu Porównawczego Transkrypcji ---")

    results = {}

    # --- Test 1: Model z config.ini (domyślnie 'small') ---
    settings_small = load_configuration()
    model_small = load_model(settings_small)
    if model_small:
        results['raw_small'] = transcribe_audio(model_small, RAW_AUDIO_PATH, settings_small['language'])
        results['processed_small'] = transcribe_audio(model_small, PROCESSED_AUDIO_PATH, settings_small['language'])
        del model_small

    # --- Test 2: Model 'medium' ---
    settings_medium = load_configuration(override_model_path="medium")
    model_medium = load_model(settings_medium)
    if model_medium:
        results['raw_medium'] = transcribe_audio(model_medium, RAW_AUDIO_PATH, settings_medium['language'])
        results['processed_medium'] = transcribe_audio(model_medium, PROCESSED_AUDIO_PATH, settings_medium['language'])
        del model_medium

    # --- Podsumowanie Wyników ---
    print("\n\n" + "="*80)
    print("--- PODSUMOWANIE WYNIKÓW TESTU PORÓWNAWCZEGO ---")
    print("="*80)

    print("\n--- 1. SUROWY PLIK (.wav) ---")
    print(f"\n[Model 'small']:\n{results.get('raw_small', 'N/A')}")
    print(f"\n[Model 'medium']:\n{results.get('raw_medium', 'N/A')}")

    print("\n\n" + "-"*80)

    print("\n--- 2. PLIK PRZETWORZONY (po preprocessingu) ---")
    print(f"\n[Model 'small']:\n{results.get('processed_small', 'N/A')}")
    print(f"\n[Model 'medium']:\n{results.get('processed_medium', 'N/A')}")

    print("\n" + "="*80)
    print("--- Koniec Testu ---")

if __name__ == "__main__":
    main()