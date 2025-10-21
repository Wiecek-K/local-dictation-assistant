# /testing_suite/run_comparison_test.py
# Wersja 8: Naprawiono błąd krytyczny. Wartość 'auto' dla języka
# jest teraz poprawnie konwertowana na None przed przekazaniem do modelu.

import sys
import time
import configparser
import librosa
from faster_whisper import WhisperModel
import os
import numpy as np

try:
    import nvsmi
except ImportError:
    print("Biblioteka nvsmi nie jest zainstalowana. Uruchom 'pip install nvsmi'.")
    nvsmi = None

# --- Konfiguracja Ścieżek i Importów ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)
from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE

CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')
RAW_AUDIO_PATH = os.path.join(PARENT_DIR, 'sibilants_test.wav')

def get_gpu_usage():
    if not nvsmi: return "N/A"
    try:
        gpu = list(nvsmi.get_gpus())[0]
        return f"{gpu.mem_util}% ({gpu.mem_used}MB / {gpu.mem_total}MB)"
    except Exception: return "Błąd odczytu GPU"

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

def transcribe_audio(model, audio_data, language, label=""):
    """Transkrybuje dane audio (numpy array) i zwraca tekst oraz czas."""
    print(f"   -> Transkrypcja danych: {label}...")
    try:
        start_time = time.time()
        # --- POPRAWKA TUTAJ ---
        # Konwertuj 'auto' na None, aby włączyć automatyczne wykrywanie języka.
        language_for_model = None if str(language).lower() == 'auto' else language
        
        segments_generator, _ = model.transcribe(audio_data, language=language_for_model, beam_size=5)
        text_result = "".join(segment.text for segment in segments_generator).strip()
        duration = time.time() - start_time
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

    if not os.path.exists(RAW_AUDIO_PATH):
        print(f"❌ BŁĄD KRYTYCZNY: Brak pliku wejściowego: {RAW_AUDIO_PATH}")
        sys.exit(1)
    
    print(f"\n--- Wczytywanie i przetwarzanie pliku: {os.path.basename(RAW_AUDIO_PATH)} ---")
    raw_audio_data, sr = librosa.load(RAW_AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    processed_audio_data = apply_preprocessing_pipeline(raw_audio_data.copy())

    # --- Test 1: Model 'small' (wymuszony) ---
    settings_small = load_configuration(override_model_path="small")
    model_small = load_model(settings_small)
    if model_small:
        text, duration = transcribe_audio(model_small, raw_audio_data, settings_small['language'], label="Surowe audio")
        results['raw_small'] = {'text': text, 'time': duration}
        text, duration = transcribe_audio(model_small, processed_audio_data, settings_small['language'], label="Przetworzone audio")
        results['processed_small'] = {'text': text, 'time': duration}
        del model_small
        print(f"   -> Zużycie VRAM po zwolnieniu 'small': {get_gpu_usage()}")

    # --- Test 2: Model 'medium' (wymuszony) ---
    settings_medium = load_configuration(override_model_path="medium")
    model_medium = load_model(settings_medium)
    if model_medium:
        text, duration = transcribe_audio(model_medium, raw_audio_data, settings_medium['language'], label="Surowe audio")
        results['raw_medium'] = {'text': text, 'time': duration}
        text, duration = transcribe_audio(model_medium, processed_audio_data, settings_medium['language'], label="Przetworzone audio")
        results['processed_medium'] = {'text': text, 'time': duration}
        del model_medium
        print(f"   -> Zużycie VRAM po zwolnieniu 'medium': {get_gpu_usage()}")

    # --- Podsumowanie Wyników ---
    print("\n\n" + "="*80)
    print("--- PODSUMOWANIE WYNIKÓW TESTU PORÓWNAWCZEGO ---")
    print("="*80)
    def print_result(label, result_key):
        res = results.get(result_key)
        if res: print(f"\n[{label}] (czas: {res['time']:.2f}s):\n{res['text']}")
        else: print(f"\n[{label}]:\nN/A")
    print("\n--- 1. SUROWE AUDIO ---")
    print_result("Model 'small'", 'raw_small')
    print_result("Model 'medium'", 'raw_medium')
    print("\n\n" + "-"*80)
    print("\n--- 2. AUDIO PRZETWORZONE (po preprocessingu) ---")
    print_result("Model 'small'", 'processed_small')
    print_result("Model 'medium'", 'processed_medium')
    print("\n" + "="*80)
    print("--- Koniec Testu ---")

if __name__ == "__main__":
    main()