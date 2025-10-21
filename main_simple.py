# main_simple.py
# Wersja 3.5: Poprawiono nazwy przycisków bocznych myszy (x1 -> button8, x2 -> button9).

import os
from datetime import datetime
import configparser
import sys
import time
import numpy as np
import noisereduce as nr
import sounddevice as sd
import pyperclip
import subprocess
import threading
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
from pynput import keyboard, mouse
from scipy.io.wavfile import write

# --- Globalne Zmienne i Parametry (bez zmian) ---
model = None
is_recording = False
audio_frames = []
app_settings = {}
recording_stop_time = 0
SAMPLE_RATE = 16000
DEESSER_THRESH_DB = -43
DEESSER_ATTENUATION_DB = 13
DEESSER_FREQ_START = 6000
DEESSER_FREQ_END = 10000
DEESSER_ATTACK_MS = 10
DEESSER_RELEASE_MS = 30
FINAL_GAIN_DB = 6.0

# --- Funkcje Przetwarzania i Aplikacji (bez zmian) ---
def dynamic_de_esser_smooth(audio_segment, threshold_db, freq_start, freq_end, attenuation_db, attack_ms, release_ms):
    sibilance_band = audio_segment.high_pass_filter(freq_start).low_pass_filter(freq_end)
    chunk_length_ms = 10
    is_attenuating = False
    processed_audio = AudioSegment.empty()
    for i in range(0, len(audio_segment), chunk_length_ms):
        chunk_original = audio_segment[i:i+chunk_length_ms]
        chunk_sibilance = sibilance_band[i:i+chunk_length_ms]
        should_attenuate = chunk_sibilance.dBFS > threshold_db
        if should_attenuate and not is_attenuating:
            chunk_attenuated = chunk_original - attenuation_db
            transition = chunk_original.fade(to_gain=-120, start=0, duration=attack_ms).overlay(chunk_attenuated.fade(from_gain=-120, start=0, duration=attack_ms))
            processed_audio += transition
            is_attenuating = True
        elif not should_attenuate and is_attenuating:
            chunk_attenuated = chunk_original - attenuation_db
            transition = chunk_attenuated.fade(to_gain=-120, start=0, duration=release_ms).overlay(chunk_original.fade(from_gain=-120, start=0, duration=release_ms))
            processed_audio += transition
            is_attenuating = False
        elif is_attenuating:
            processed_audio += (chunk_original - attenuation_db)
        else:
            processed_audio += chunk_original
    return processed_audio

def apply_preprocessing_pipeline(audio_data_float32):
    print("🔊 Uruchamianie potoku przetwarzania wstępnego audio...")
    pipeline_start_time = time.time()
    last_step_time = pipeline_start_time
    try:
        audio_data_int16 = np.int16(audio_data_float32 * 32767)
        audio_segment = AudioSegment(
            audio_data_int16.tobytes(), 
            frame_rate=SAMPLE_RATE,
            sample_width=audio_data_int16.dtype.itemsize, 
            channels=1
        )
        print(f"   - Krok 1: Normalizacja głośności...")
        normalized_segment = normalize(audio_segment)
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time
        print(f"   - Krok 2: Aplikowanie de-essera z wygładzaniem...")
        deessed_segment = dynamic_de_esser_smooth(
            normalized_segment, 
            DEESSER_THRESH_DB, DEESSER_FREQ_START, DEESSER_FREQ_END, 
            DEESSER_ATTENUATION_DB, DEESSER_ATTACK_MS, DEESSER_RELEASE_MS
        )
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time
        print(f"   - Krok 3: Podbicie głośności o +{FINAL_GAIN_DB} dB...")
        boosted_segment = deessed_segment + FINAL_GAIN_DB
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time
        processed_before_nr = np.array(boosted_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
        print("   - Krok 4: Aplikowanie redukcji szumu...")
        noise_clip = processed_before_nr[:int(SAMPLE_RATE * 0.5)]
        final_audio_float32 = nr.reduce_noise(
            y=processed_before_nr, y_noise=noise_clip, sr=SAMPLE_RATE, prop_decrease=0.85
        )
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time
        print(f"🔊 Przetwarzanie wstępne zakończone pomyślnie (całkowity czas: {time.time() - pipeline_start_time:.2f}s).")
        return final_audio_float32
    except Exception as e:
        print(f"⚠️ OSTRZEŻENIE: Przetwarzanie wstępne nie powiodło się: {e}.")
        print("   Używanie oryginalnego, surowego audio.")
        return audio_data_float32

def load_configuration():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='medium'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
            'hotkey': config.get('settings', 'hotkey', fallback='<ctrl>+f8'),
            'language': config.get('settings', 'language', fallback='auto'),
        }
        print("Konfiguracja załadowana pomyślnie.")
        return settings
    except (FileNotFoundError, configparser.Error) as e:
        print(f"Błąd wczytywania config.ini: {e}"), sys.exit(1)

def load_model(settings):
    global model
    model_path = settings['model_path']
    device = settings['device']
    compute_type = settings['compute_type']
    print("\n--- Ładowanie Modelu ---")
    print(f"Próba załadowania modelu: '{model_path}' ({device}, {compute_type})")
    start_time = time.time()
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        print(f"✅ Model załadowany pomyślnie w {time.time() - start_time:.2f}s.")
    except Exception as e:
        print(f"❌ BŁĄD KRYTYCZNY: Nie udało się załadować modelu Whisper: {e}"), sys.exit(1)

def record_and_transcribe(settings):
    global audio_frames, recording_stop_time
    print("\n🎙️  Nagrywanie... Mów teraz.")
    audio_frames = []
    def audio_callback(indata, frames, time, status):
        if status: print(f"Status strumienia audio: {status}", file=sys.stderr)
        audio_frames.append(indata.copy())
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    while is_recording:
        time.sleep(0.1)
    stream.stop()
    stream.close()
    print("🎙️  Nagrywanie zatrzymane.")
    if not audio_frames:
        print("Nie nagrano żadnego dźwięku.")
        return
    raw_audio_data = np.concatenate(audio_frames, axis=0).flatten().astype(np.float32)
    
    processed_audio = apply_preprocessing_pipeline(raw_audio_data)
    audio_duration_seconds = len(processed_audio) / SAMPLE_RATE
    
    lang_setting = settings['language']
    language_for_model = None if lang_setting.lower() == 'auto' else lang_setting
    
    if language_for_model is None:
        print("🧠 Rozpoczynanie transkrypcji (z automatycznym wykrywaniem języka)...")
    else:
        print(f"🧠 Rozpoczynanie transkrypcji (język: {language_for_model})...")
        
    print(f"   -> Długość audio do transkrypcji: {audio_duration_seconds:.2f}s")
    
    transcription_start_time = time.time()
    segments_generator, info = model.transcribe(
        processed_audio, language=language_for_model, beam_size=5
    )
    
    if language_for_model is None:
        print(f"   -> Wykryto język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

    final_text = "".join(segment.text for segment in segments_generator).strip()
    transcription_end_time = time.time()
    
    transcription_duration = transcription_end_time - transcription_start_time
    
    print("\n--- Wynik Końcowy ---")
    print(f"Tekst: {final_text}")
    if final_text:
        before_clipboard_time = time.time()
        pyperclip.copy(final_text)
        after_clipboard_time = time.time()
        print("✅ Skopiowano do schowka.")
        before_paste_time = time.time()
        try:
            time.sleep(0.1)
            subprocess.run(["xdotool", "type", "--delay", "1", "--clearmodifiers", final_text], check=True)
            pasting_end_time = time.time()
            print("✅ Wklejono do aktywnego okna.")
            time_to_transcribe = transcription_end_time - recording_stop_time
            time_to_paste = pasting_end_time - recording_stop_time
            clipboard_duration = after_clipboard_time - before_clipboard_time
            pasting_duration = pasting_end_time - before_paste_time
            rtf = float('inf')
            if audio_duration_seconds > 0:
                rtf = transcription_duration / audio_duration_seconds
            print("\n--- Statystyki Czasowe ---")
            print(f"🎧 Długość nagrania: {audio_duration_seconds:.2f}s")
            print(f"🧠 Czas samej transkrypcji (z iteracją): {transcription_duration:.2f}s")
            print(f"🚀 Współczynnik RTF (Real-Time Factor): {rtf:.3f}")
            print(f"⏱️ Czas do transkrypcji (od puszczenia klawisza): {time_to_transcribe:.2f}s")
            print(f"📋 Czas kopiowania do schowka (pyperclip): {clipboard_duration:.2f}s")
            print(f"⌨️ Czas samego wklejania (xdotool + sleep): {pasting_duration:.2f}s")
            print(f"⏱️ Czas do wklejenia (całkowity czas oczekiwania): {time_to_paste:.2f}s")
        except FileNotFoundError:
            print("❌ BŁĄD: Polecenie 'xdotool' nie zostało znalezione.")
        except Exception as e:
            print(f"❌ Błąd podczas wklejania tekstu: {e}")

def start_recording_flag():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_and_transcribe, args=(app_settings,)).start()

def stop_recording_flag():
    global is_recording, recording_stop_time
    if is_recording:
        recording_stop_time = time.time()
        is_recording = False

# --- ZMIANA TUTAJ: Poprawiony parser hotkey ---
def parse_hotkey(hotkey_string):
    hotkey_string = hotkey_string.lower().strip()

    if hotkey_string.startswith('mouse:'):
        button_name = hotkey_string.split(':')[1].strip()
        # --- POPRAWKA TUTAJ: Użyto poprawnych nazw przycisków ---
        button_map = {
            'button4': mouse.Button.button8, # Zazwyczaj "wstecz"
            'button5': mouse.Button.button9, # Zazwyczaj "do przodu"
            'left': mouse.Button.left,
            'right': mouse.Button.right,
            'middle': mouse.Button.middle,
        }
        if button_name in button_map:
            return {'type': 'mouse', 'button': button_map[button_name]}
        else:
            print(f"⚠️ OSTRZEŻENIE: Nieznany przycisk myszy w config.ini: '{button_name}'")
            return None
    else:
        keys = set()
        parts = hotkey_string.split('+')
        for part in parts:
            part = part.strip()
            key_name = part
            if part.startswith('<') and part.endswith('>'):
                key_name = part[1:-1]
            try:
                key = getattr(keyboard.Key, key_name)
                keys.add(key)
            except AttributeError:
                if len(key_name) == 1:
                    keys.add(keyboard.KeyCode.from_char(key_name))
                else:
                    print(f"⚠️ OSTRZEŻENIE: Nieznany klawisz w config.ini: '{key_name}'")
        return {'type': 'keyboard', 'key_set': keys}

# --- Główna pętla z wyborem listenera (bez zmian) ---
if __name__ == "__main__":
    print("--- Uruchamianie Lokalnego Asystenta Dyktowania (Wersja Wsadowa) ---")
    app_settings = load_configuration()
    load_model(app_settings)
    hotkey_str = app_settings['hotkey']
    
    hotkey_config = parse_hotkey(hotkey_str)
    
    if not hotkey_config:
        print("❌ BŁĄD KRYTYCZNY: Nie udało się sparsować skrótu. Kończenie pracy.")
        sys.exit(1)

    print(f"\n✅ Gotowy. Naciśnij i przytrzymaj '{hotkey_str}', aby nagrywać. Puść, aby transkrybować.")
    print("Naciśnij Ctrl+C, aby wyjść.")
    
    if hotkey_config['type'] == 'keyboard':
        HOTKEY_COMBINATION = hotkey_config['key_set']
        current_keys = set()

        def on_press_keyboard(key):
            if key in HOTKEY_COMBINATION:
                current_keys.add(key)
                if current_keys == HOTKEY_COMBINATION:
                    start_recording_flag()

        def on_release_keyboard(key):
            if key in HOTKEY_COMBINATION:
                stop_recording_flag()
                try:
                    current_keys.remove(key)
                except KeyError:
                    pass
        
        listener = keyboard.Listener(on_press=on_press_keyboard, on_release=on_release_keyboard)

    elif hotkey_config['type'] == 'mouse':
        MOUSE_BUTTON = hotkey_config['button']

        def on_click_mouse(x, y, button, pressed):
            if button == MOUSE_BUTTON:
                if pressed:
                    start_recording_flag()
                else:
                    stop_recording_flag()

        listener = mouse.Listener(on_click=on_click_mouse)

    with listener:
        listener.join()