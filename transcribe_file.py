# transcribe_file.py
# Wersja 2.0: Przekształcono w narzędzie CLI do transkrypcji pojedynczych plików.
# Użycie: python transcribe_file.py <ścieżka_do_pliku_audio> [--no-preprocessing]

import sys
import os
import time
import argparse
import configparser
import librosa
from faster_whisper import WhisperModel

# Dodaj katalog główny do ścieżki, aby umożliwić import audio_processing
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
try:
    from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE
except ImportError:
    print("BŁĄD: Nie można zaimportować modułu 'audio_processing'. Upewnij się, że plik audio_processing.py istnieje.")
    sys.exit(1)

def load_configuration():
    """Wczytuje konfigurację modelu z pliku config.ini."""
    config = configparser.ConfigParser()
    config_path = os.path.join(ROOT_DIR, 'config.ini')
    try:
        config.read(config_path)
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='medium'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
            'language': config.get('settings', 'language', fallback='auto'),
        }
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
    # --- Parsowanie argumentów linii poleceń ---
    parser = argparse.ArgumentParser(
        description="Dokonuje transkrypcji pliku audio przy użyciu modelu Whisper."
    )
    parser.add_argument(
        "filepath",
        help="Ścieżka do pliku audio do przetworzenia."
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Wyłącza potok przetwarzania wstępnego audio (transkrybuje surowy plik)."
    )
    args = parser.parse_args()

    # --- Walidacja pliku wejściowego ---
    if not os.path.exists(args.filepath):
        print(f"❌ BŁĄD: Plik nie istnieje: {args.filepath}")
        sys.exit(1)

    # --- Wczytanie konfiguracji i modelu ---
    app_settings = load_configuration()
    model = load_model(app_settings)

    # --- Wczytanie i przetwarzanie audio ---
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

    # --- Transkrypcja ---
    print("\n🧠 Rozpoczynanie transkrypcji...")
    transcription_start_time = time.time()
    
    language_for_model = None if app_settings['language'].lower() == 'auto' else app_settings['language']
    
    segments_generator, info = model.transcribe(
        audio_data, language=language_for_model, beam_size=5
    )

    if language_for_model is None:
        print(f"   -> Wykryto język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    final_text = "".join(segment.text for segment in segments_generator).strip()
    transcription_duration = time.time() - transcription_start_time
    
    print(f"   -> Transkrypcja zakończona w {transcription_duration:.2f}s.")

    # --- Wyświetlenie wyniku ---
    print("\n" + "="*80)
    print("--- WYNIK TRANSKRYPCJI ---")
    print("="*80)
    print(final_text)
    print("="*80)

if __name__ == "__main__":
    main()