# main_simple.py
# Wersja 4.1: Usunięto logikę obliczania skuteczności VAD.

import configparser
import sys
import time
import numpy as np
import sounddevice as sd
import pyperclip
import subprocess
import threading
from faster_whisper import WhisperModel
from pynput import keyboard, mouse
import logging

from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE
from src.logger_setup import setup_loggers
from src.core_utils import load_configuration, load_model

# --- Inicjalizacja Loggerów ---
app_logger = logging.getLogger('app')
transcription_logger = logging.getLogger('transcription')
performance_logger = logging.getLogger('performance')

# --- Globalne Zmienne i Parametry ---
model = None
is_recording = False
audio_frames = []
app_settings = {}
recording_stop_time = 0

def record_and_transcribe(settings, model_instance):
    global audio_frames, recording_stop_time
    app_logger.info("\n🎙️  Nagrywanie... Mów teraz.")
    audio_frames = []
    def audio_callback(indata, frames, time, status):
        if status: app_logger.warning(f"Status strumienia audio: {status}", file=sys.stderr)
        audio_frames.append(indata.copy())
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    while is_recording:
        time.sleep(0.1)
    stream.stop()
    stream.close()
    app_logger.info("🎙️  Nagrywanie zatrzymane.")
    if not audio_frames:
        app_logger.warning("Nie nagrano żadnego dźwięku.")
        return
    raw_audio_data = np.concatenate(audio_frames, axis=0).flatten().astype(np.float32)
    
    processed_audio = apply_preprocessing_pipeline(raw_audio_data)
    original_duration_seconds = len(processed_audio) / SAMPLE_RATE
    
    lang_setting = settings['language']
    language_for_model = None if lang_setting.lower() == 'auto' else lang_setting
    
    transcription_logger.info("🧠 Rozpoczynanie transkrypcji...")
    transcription_logger.info(f"   -> Długość audio (po preprocessingu): {original_duration_seconds:.2f}s")
    transcription_logger.debug(f"   -> Używane parametry: VAD={settings['vad_filter']}, LogProb={settings['log_prob_threshold']}, NoSpeech={settings['no_speech_threshold']}")
    
    transcription_start_time = time.time()
    
    segments_generator, info = model_instance.transcribe(
        processed_audio,
        language=language_for_model,
        beam_size=5,
        vad_filter=settings['vad_filter'],
        log_prob_threshold=settings['log_prob_threshold'],
        no_speech_threshold=settings['no_speech_threshold']
    )
    
    if language_for_model is None:
        transcription_logger.info(f"   -> Wykryto język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    # --- ZMIANA TUTAJ: Usunięto konwersję na listę i obliczenia VAD ---
    final_text = "".join(seg.text for seg in segments_generator).strip() + " "
    transcription_end_time = time.time()
    
    transcription_duration = transcription_end_time - transcription_start_time
    
    app_logger.info("\n--- Wynik Końcowy ---")
    app_logger.info(f"Tekst: {final_text}")
    if final_text.strip():
        before_clipboard_time = time.time()
        pyperclip.copy(final_text)
        after_clipboard_time = time.time()
        app_logger.info("✅ Skopiowano do schowka.")
        before_paste_time = time.time()
        try:
            time.sleep(0.1)
            subprocess.run(["xdotool", "type", "--delay", "1", "--clearmodifiers", final_text], check=True)
            pasting_end_time = time.time()
            app_logger.info("✅ Wklejono do aktywnego okna.")
            time_to_transcribe = transcription_end_time - recording_stop_time
            time_to_paste = pasting_end_time - recording_stop_time
            clipboard_duration = after_clipboard_time - before_clipboard_time
            pasting_duration = pasting_end_time - before_paste_time
            rtf = float('inf')
            if original_duration_seconds > 0:
                rtf = transcription_duration / original_duration_seconds
            performance_logger.info("\n--- Statystyki Czasowe ---")
            performance_logger.info(f"🎧 Długość nagrania: {original_duration_seconds:.2f}s")
            performance_logger.info(f"🧠 Czas samej transkrypcji (z iteracją): {transcription_duration:.2f}s")
            performance_logger.info(f"🚀 Współczynnik RTF (Real-Time Factor): {rtf:.3f}")
            performance_logger.info(f"⏱️ Czas do transkrypcji (od puszczenia klawisza): {time_to_transcribe:.2f}s")
            performance_logger.info(f"📋 Czas kopiowania do schowka (pyperclip): {clipboard_duration:.2f}s")
            performance_logger.info(f"⌨️ Czas samego wklejania (xdotool + sleep): {pasting_duration:.2f}s")
            performance_logger.info(f"⏱️ Czas do wklejenia (całkowity czas oczekiwania): {time_to_paste:.2f}s")
        except FileNotFoundError:
            app_logger.error("❌ BŁĄD: Polecenie 'xdotool' nie zostało znalezione.")
        except Exception as e:
            app_logger.error(f"❌ Błąd podczas wklejania tekstu: {e}")

def start_recording_flag(model_instance): # ZMIANA: Funkcja przyjmuje model jako argument
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_and_transcribe, args=(app_settings, model_instance)).start() 

def stop_recording_flag():
    global is_recording, recording_stop_time
    if is_recording:
        recording_stop_time = time.time()
        is_recording = False

def parse_hotkey(hotkey_string):
    hotkey_string = hotkey_string.lower().strip()
    if hotkey_string.startswith('mouse:'):
        button_name = hotkey_string.split(':')[1].strip()
        button_map = {
            'button4': mouse.Button.button8, 'button5': mouse.Button.button9,
            'left': mouse.Button.left, 'right': mouse.Button.right, 'middle': mouse.Button.middle,
        }
        if button_name in button_map:
            return {'type': 'mouse', 'button': button_map[button_name]}
        else:
            app_logger.warning(f"⚠️ OSTRZEŻENIE: Nieznany przycisk myszy w config.ini: '{button_name}'")
            return None
    else:
        keys = set()
        parts = hotkey_string.split('+')
        for part in parts:
            part = part.strip()
            key_name = part[1:-1] if part.startswith('<') and part.endswith('>') else part
            try:
                key = getattr(keyboard.Key, key_name)
                keys.add(key)
            except AttributeError:
                if len(key_name) == 1:
                    keys.add(keyboard.KeyCode.from_char(key_name))
                else:
                    app_logger.warning(f"⚠️ OSTRZEŻENIE: Nieznany klawisz w config.ini: '{key_name}'")
        return {'type': 'keyboard', 'key_set': keys}

if __name__ == "__main__":
    setup_loggers()
    
    app_logger.info("--- Uruchamianie Lokalnego Asystenta Dyktowania (Wersja Wsadowa) ---")
    app_settings = load_configuration()
    model_instance = load_model(app_settings) 
    hotkey_str = app_settings['hotkey']
    hotkey_config = parse_hotkey(hotkey_str)
    
    if not hotkey_config:
        app_logger.critical("❌ BŁĄD KRYTYCZNY: Nie udało się sparsować skrótu. Kończenie pracy.")
        sys.exit(1)

    app_logger.info(f"\n✅ Gotowy. Naciśnij i przytrzymaj '{hotkey_str}', aby nagrywać. Puść, aby transkrybować.")
    app_logger.info("Naciśnij Ctrl+C, aby wyjść.")
    
    listener = None
    if hotkey_config['type'] == 'keyboard':
        HOTKEY_COMBINATION = hotkey_config['key_set']
        current_keys = set()
        def on_press_keyboard(key):
            if key in HOTKEY_COMBINATION:
                current_keys.add(key)
                if current_keys == HOTKEY_COMBINATION: start_recording_flag(model_instance) 
        def on_release_keyboard(key):
            if key in HOTKEY_COMBINATION:
                stop_recording_flag()
                try: current_keys.remove(key)
                except KeyError: pass
        listener = keyboard.Listener(on_press=on_press_keyboard, on_release=on_release_keyboard)
    elif hotkey_config['type'] == 'mouse':
        MOUSE_BUTTON = hotkey_config['button']
        def on_click_mouse(x, y, button, pressed):
            if button == MOUSE_BUTTON:
                if pressed: start_recording_flag(model_instance) 
                else: stop_recording_flag()
        listener = mouse.Listener(on_click=on_click_mouse)

    if listener:
        with listener:
            listener.join()