# /testing_suite/run_comparison_test.py
# Wersja 4: Dodano logowanie zużycia VRAM po załadowaniu każdego modelu.

import sys
import time
import configparser
import librosa
from faster_whisper import WhisperModel
import os
try:
    import nvsmi
except ImportError:
    print("Biblioteka nvsmi nie jest zainstalowana. Uruchom 'pip install nvsmi'.")
    nvsmi = None

# --- Konfiguracja Ścieżek ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')
RAW_AUDIO_PATH = os.path.join(PARENT_DIR, 'sibilants_test.wav')
PROCESSED_AUDIO_PATH = os.path.join(PARENT_DIR, 'test_outputs', 'sibilants_test_processed.wav')

def get_gpu_usage():
    """Zwraca informacje o zużyciu VRAM, jeśli nvsmi jest dostępne."""
    if not nvsmi:
        return "N/A (biblioteka nvsmi nie jest zainstalowana)"
    try:
        gpus = list(nvsmi.get_gpus())
        if not gpus:
            return "Nie znaleziono karty NVIDIA."
        # Zakładamy, że interesuje nas pierwsza karta
        gpu = gpus[0]
        return f"{gpu.mem_util}% ({gpu.mem_used}MB / {gpu.mem_total}MB)"
    except Exception as e:
        return f"Błąd odczytu informacji o GPU: {e}"

def load_configuration(override_model_path=None):
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
            settings['model_path'] = override_model_path
        return settings
    except Exception as e:
        print(f"Błąd wczytywania {CONFIG_PATH}: {e}"), sys.exit(1)

def load_model(settings):
    model_path = settings['model_path']
    print(f"\n--- Ładowanie Modelu '{model_path}' ---")
    start_time = time.time()
    try:
        model = WhisperModel(model_path, device=settings['device'], compute_type=settings['compute_type'])
        print(f"✅ Model załadowany w {time.time() - start_time:.2f}s.")
        print(f"   -> Zużycie VRAM po załadowaniu: {get_gpu_usage()}")
        return model
    except Exception as e:
        print(f"❌ BŁĄD KRYTYCZNY: {e}"),
        return None

def transcribe_audio(model, file_path, language):
    if not os.path.exists(file_path):
        error_msg = f"[BŁĄD: BRAK PLIKU: {os.path.basename(file_path)}]"
        print(f"❌ {error_msg}")
        return error_msg, 0.0
    print(f"   -> Transkrypcja pliku: {os.path.basename(file_path)}...")
    try:
        audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
        start_time = time.time()
        segments, _ = model.transcribe(audio_data, language=language, beam_size=5)
        duration = time.time() - start_time
        text_result = "".join(segment.text for segment in segments).strip()
        print(f"   -> Zakończono w {duration:.2f}s.")
        return text_result, duration
    except Exception as e:
        error_msg = f"[BŁĄD TRANSKRYPCJI: {e}]"
        print(f"   -> ❌ {error_msg}")
        return error_msg, 0.0

def main():
    print("--- Uruchamianie Pełnego Testu Porównawczego Transkrypcji ---")
    results = {}

    print(f"Początkowe zużycie VRAM: {get_gpu_usage()}")

    settings_small = load_configuration()
    model_small = load_model(settings_small)
    if model_small:
        text, duration = transcribe_audio(model_small, RAW_AUDIO_PATH, settings_small['language'])
        results['raw_small'] = {'text': text, 'time': duration}
        text, duration = transcribe_audio(model_small, PROCESSED_AUDIO_PATH, settings_small['language'])
        results['processed_small'] = {'text': text, 'time': duration}
        del model_small
        print("   -> Model 'small' zwolniony z pamięci.")
        print(f"   -> Zużycie VRAM po zwolnieniu: {get_gpu_usage()}")

    settings_medium = load_configuration(override_model_path="medium")
    model_medium = load_model(settings_medium)
    if model_medium:
        text, duration = transcribe_audio(model_medium, RAW_AUDIO_PATH, settings_medium['language'])
        results['raw_medium'] = {'text': text, 'time': duration}
        text, duration = transcribe_audio(model_medium, PROCESSED_AUDIO_PATH, settings_medium['language'])
        results['processed_medium'] = {'text': text, 'time': duration}
        del model_medium
        print("   -> Model 'medium' zwolniony z pamięci.")
        print(f"   -> Zużycie VRAM po zwolnieniu: {get_gpu_usage()}")

    # --- Podsumowanie Wyników ---
    print("\n\n" + "="*80)
    print("--- PODSUMOWANIE WYNIKÓW TESTU PORÓWNAWCZEGO ---")
    print("="*80)
    def print_result(label, result_key):
        res = results.get(result_key)
        if res: print(f"\n[{label}] (czas: {res['time']:.2f}s):\n{res['text']}")
        else: print(f"\n[{label}]:\nN/A")
    print("\n--- 1. SUROWY PLIK (.wav) ---")
    print_result("Model 'small'", 'raw_small')
    print_result("Model 'medium'", 'raw_medium')
    print("\n\n" + "-"*80)
    print("\n--- 2. PLIK PRZETWORZONY (po preprocessingu) ---")
    print_result("Model 'small'", 'processed_small')
    print_result("Model 'medium'", 'processed_medium')
    print("\n" + "="*80)
    print("--- Koniec Testu ---")

if __name__ == "__main__":
    main()