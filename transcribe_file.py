# transcribe_file.py
# Wersja 2.1: Zaktualizowano wczytywanie konfiguracji z podziałem na sekcje
# i dodano obsługę zaawansowanych parametrów transkrypcji.

import sys
import os
import time
import argparse
import configparser
import librosa
from faster_whisper import WhisperModel

# Dodaj katalog główny do ścieżki, aby umożliwić import
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
try:
    from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE
except ImportError:
    print("BŁĄD: Nie można zaimportować modułu. Upewnij się, że plik src/audio_preprocessing.py istnieje.")
    sys.exit(1)

def load_configuration():
    """Wczytuje konfigurację z pliku config.ini z podziałem na sekcje."""
    config = configparser.ConfigParser()
    config_path = os.path.join(ROOT_DIR, 'config.ini')
    try:
        config.read(config_path)
        # Wczytaj ustawienia podstawowe
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='medium'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'language': config.get('settings', 'language', fallback='auto'),
        }
        # Wczytaj ustawienia zaawansowane i dokonaj konwersji typów
        settings.update({
            'compute_type': config.get('advanced', 'compute_type', fallback='int8'),
            'vad_filter': config.getboolean('advanced', 'vad_filter', fallback=True),
            'log_prob_threshold': config.getfloat('advanced', 'log_prob_threshold', fallback=-1.0),
            'no_speech_threshold': config.getfloat('advanced', 'no_speech_threshold', fallback=0.6),
        })
        return settings
    except Exception as e:
        print(f"Błąd wczytywania {config_path}: {e}")
        sys.exit(1)

def load_model(settings):
    """Wczytuje i zwraca model Whisper na podstawie ustawień."""
    model_path = settings['model_path']
    print(f"--- Ładowanie Modelu '{model_path}' ({settings['device']}, {settings['compute_type']}) ---")
    start_time = time.time()
    try:
        model = WhisperModel(model_path, device=settings['device'], compute_type=settings['compute_type'])
        print(f"✅ Model załadowany w {time.time() - start_time:.2f}s.")
        return model
    except Exception as e:
        print(f"❌ BŁĄD KRYTYCZNY: Nie udało się załadować modelu: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Dokonuje transkrypcji pliku audio przy użyciu modelu Whisper."
    )
    parser.add_argument("filepath", help="Ścieżka do pliku audio do przetworzenia.")
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Wyłącza potok przetwarzania wstępnego audio."
    )
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"❌ BŁĄD: Plik nie istnieje: {args.filepath}")
        sys.exit(1)

    app_settings = load_configuration()
    model = load_model(app_settings)

    print(f"\n--- Przetwarzanie pliku: {os.path.basename(args.filepath)} ---")
    try:
        audio_data, _ = librosa.load(args.filepath, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"❌ BŁĄD: Nie udało się wczytać pliku audio: {e}")
        sys.exit(1)

    if not args.no_preprocessing:
        audio_data = apply_preprocessing_pipeline(audio_data)
    else:
        print("🔊 Przetwarzanie wstępne audio pominięte (opcja --no-preprocessing).")

    print("\n🧠 Rozpoczynanie transkrypcji...")
    transcription_start_time = time.time()
    
    language_for_model = None if app_settings['language'].lower() == 'auto' else app_settings['language']
    
    # --- ZMIANA TUTAJ: Przekazanie nowych parametrów do modelu ---
    segments_generator, info = model.transcribe(
        audio_data,
        language=language_for_model,
        beam_size=5,
        vad_filter=app_settings['vad_filter'],
        log_prob_threshold=app_settings['log_prob_threshold'],
        no_speech_threshold=app_settings['no_speech_threshold']
    )

    if language_for_model is None:
        print(f"   -> Wykryto język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    final_text = "".join(segment.text for segment in segments_generator).strip()
    transcription_duration = time.time() - transcription_start_time
    
    print(f"   -> Transkrypcja zakończona w {transcription_duration:.2f}s.")

    print("\n" + "="*80)
    print("--- WYNIK TRANSKRYPCJI ---")
    print("="*80)
    print(final_text)
    print("="*80)

if __name__ == "__main__":
    main()