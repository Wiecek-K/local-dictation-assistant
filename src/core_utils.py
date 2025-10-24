# FILE: src/core_utils.py

import configparser
import sys
import time
import logging
from faster_whisper import WhisperModel
import os

# --- Inicjalizacja Loggerów ---
app_logger = logging.getLogger('app')

def load_configuration():
    """
    Wczytuje konfigurację z pliku config.ini z podziałem na sekcje.
    Obsługuje ścieżki względne z poziomu głównego katalogu projektu.
    """
    config = configparser.ConfigParser()
    # Ścieżka do config.ini jest względna do głównego katalogu projektu
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
    
    try:
        config.read(config_path)
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='medium'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'hotkey': config.get('settings', 'hotkey', fallback='<ctrl>+f8'),
            'language': config.get('settings', 'language', fallback='auto'),
        }
        settings.update({
            'compute_type': config.get('advanced', 'compute_type', fallback='int8'),
            'vad_filter': config.getboolean('advanced', 'vad_filter', fallback=True),
            'log_prob_threshold': config.getfloat('advanced', 'log_prob_threshold', fallback=-1.0),
            'no_speech_threshold': config.getfloat('advanced', 'no_speech_threshold', fallback=0.6),
            'local_files_only': config.getboolean('advanced', 'local_files_only', fallback=True),
            'beam_size': config.getint('advanced', 'beam_size', fallback=5),
            'compression_ratio_threshold': config.getfloat('advanced', 'compression_ratio_threshold', fallback=3.0), # ZMIANA
            # NOWE PARAMETRY VAD
            'vad_max_buffer_seconds': config.getint('advanced', 'vad_max_buffer_seconds', fallback=20),
            'vad_min_chunk_seconds': config.getint('advanced', 'vad_min_chunk_seconds', fallback=10),
            'vad_silence_threshold_seconds': config.getfloat('advanced', 'vad_silence_threshold_seconds', fallback=1.5),
            'vad_rms_threshold': config.getfloat('advanced', 'vad_rms_threshold', fallback=0.005)
        })
        # USUNIĘTO: streaming_vad_mode
        app_logger.info("Konfiguracja załadowana pomyślnie.")
        return settings
    except (FileNotFoundError, configparser.Error) as e:
        app_logger.error(f"Błąd wczytywania config.ini: {e}")
        sys.exit(1)

def load_model(settings):
    """Wczytuje i zwraca model Whisper na podstawie ustawień."""
    app_logger.info("\n--- Ładowanie Modelu ---")
    app_logger.info(f"Próba załadowania modelu: '{settings['model_path']}' ({settings['device']}, {settings['compute_type']})")
    start_time = time.time()
    try:
        model = WhisperModel(
            settings['model_path'], 
            device=settings['device'], 
            compute_type=settings['compute_type'], 
            local_files_only=settings['local_files_only']
        )
        app_logger.info(f"✅ Model załadowany pomyślnie w {time.time() - start_time:.2f}s.")
        return model
    except Exception as e:
        app_logger.critical(f"❌ BŁĄD KRYTYCZNY: Nie udało się załadować modelu Whisper: {e}")
        sys.exit(1)