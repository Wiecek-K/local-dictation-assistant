# main_simple.py
# Wersja 3.8: Dodano spację na końcu transkrypcji w celu ułatwienia dyktowania seryjnego.

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

from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE

# --- Globalne Zmienne i Parametry ---
model = None                        # Przechowuje załadowaną instancję modelu Whisper.
is_recording = False                # Flaga (boolean) kontrolująca stan nagrywania (True, gdy nagrywa).
audio_frames = []                   # Lista przechowująca fragmenty (ramki) surowego audio podczas nagrywania.
app_settings = {}                   # Słownik przechowujący ustawienia wczytane z pliku config.ini.
recording_stop_time = 0             # Przechowuje znacznik czasu (timestamp) zatrzymania nagrywania do pomiaru wydajności.

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

    # --- ZMIANA TUTAJ: Dodano spację na końcu tekstu ---
    final_text = "".join(segment.text for segment in segments_generator).strip() + " "
    transcription_end_time = time.time()
    
    transcription_duration = transcription_end_time - transcription_start_time
    
    print("\n--- Wynik Końcowy ---")
    print(f"Tekst: {final_text}")
    if final_text.strip(): # Sprawdź, czy tekst nie składa się tylko ze spacji
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

def parse_hotkey(hotkey_string):
    hotkey_string = hotkey_string.lower().strip()

    if hotkey_string.startswith('mouse:'):
        button_name = hotkey_string.split(':')[1].strip()
        button_map = {
            'button4': mouse.Button.button8,
            'button5': mouse.Button.button9,
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