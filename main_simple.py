# main_simple.py
# Wersja 2.1: Dodano logowanie czasów od puszczenia klawisza do transkrypcji i wklejenia.

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
from pynput import keyboard
from scipy.io.wavfile import write

# --- Globalne Zmienne ---
model = None
is_recording = False
audio_frames = []
app_settings = {}
recording_stop_time = 0 # NOWA ZMIENNA GLOBALNA do przechowywania czasu

# --- Finalne Parametry Potoku (ustalone na podstawie testów) ---
SAMPLE_RATE = 16000             # [Hz] Częstotliwość próbkowania; standard dla modeli mowy.
DEESSER_THRESH_DB = -43         # [dB] Próg głośności, powyżej którego de-esser zaczyna tłumić syki.
DEESSER_ATTENUATION_DB = 13     # [dB] Siła, z jaką de-esser ścisza wykryte sybilanty.
DEESSER_FREQ_START = 6000       # [Hz] Dolna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_FREQ_END = 10000        # [Hz] Górna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_ATTACK_MS = 10          # [ms] Czas potrzebny na osiągnięcie pełnego tłumienia (wygładza początek).
DEESSER_RELEASE_MS = 30         # [ms] Czas powrotu do normalnej głośności (wygładza koniec, eliminuje trzaski).
FINAL_GAIN_DB = 6.0             # [dB] Końcowe podbicie głośności całego nagrania po przetworzeniu.
# DEPLOSER_FREQ = 100           # [Hz] De-ploser obecnie nieużywany, ale parametr zostaje na przyszłość.

# --- Funkcje Przetwarzania Wstępnego Audio (bez zmian) ---

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
    try:
        audio_data_int16 = np.int16(audio_data_float32 * 32767)
        audio_segment = AudioSegment(
            audio_data_int16.tobytes(), 
            frame_rate=SAMPLE_RATE,
            sample_width=audio_data_int16.dtype.itemsize, 
            channels=1
        )
        print("   - Krok 1: Normalizacja głośności...")
        normalized_segment = normalize(audio_segment)
        print(f"   - Krok 2: Aplikowanie de-essera z wygładzaniem...")
        deessed_segment = dynamic_de_esser_smooth(
            normalized_segment, 
            DEESSER_THRESH_DB, DEESSER_FREQ_START, DEESSER_FREQ_END, 
            DEESSER_ATTENUATION_DB, DEESSER_ATTACK_MS, DEESSER_RELEASE_MS
        )
        print(f"   - Krok 3: Podbicie głośności o +{FINAL_GAIN_DB} dB...")
        boosted_segment = deessed_segment + FINAL_GAIN_DB
        processed_before_nr = np.array(boosted_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
        print("   - Krok 4: Aplikowanie redukcji szumu...")
        noise_clip = processed_before_nr[:int(SAMPLE_RATE * 0.5)]
        final_audio_float32 = nr.reduce_noise(
            y=processed_before_nr, y_noise=noise_clip, sr=SAMPLE_RATE, prop_decrease=0.85
        )
        print("🔊 Przetwarzanie wstępne zakończone pomyślnie.")
        return final_audio_float32
    except Exception as e:
        print(f"⚠️ OSTRZEŻENIE: Przetwarzanie wstępne nie powiodło się: {e}.")
        print("   Używanie oryginalnego, surowego audio.")
        return audio_data_float32

# --- Główne Funkcje Aplikacji ---

def load_configuration():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='medium'), # Zaktualizowano domyślny model
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
            'hotkey': config.get('settings', 'hotkey', fallback='<ctrl>+f8'),
            'language': config.get('settings', 'language', fallback='pl'),
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
    
    # --- Transkrypcja ---
    print("🧠 Rozpoczynanie transkrypcji...")
    segments, _ = model.transcribe(
        processed_audio, language=settings['language'], beam_size=5
    )
    transcription_end_time = time.time() # ZAPISZ CZAS ZAKOŃCZENIA TRANSKRYPCJI
    final_text = "".join(segment.text for segment in segments).strip()
    
    print("\n--- Wynik Końcowy ---")
    print(f"Tekst: {final_text}")

    if final_text:
        pyperclip.copy(final_text)
        print("✅ Skopiowano do schowka.")
        try:
            time.sleep(0.1)
            subprocess.run(["xdotool", "type", "--clearmodifiers", final_text], check=True)
            pasting_end_time = time.time() # ZAPISZ CZAS ZAKOŃCZENIA WKLEJANIA
            print("✅ Wklejono do aktywnego okna.")
            
            # --- POMIAR CZASU ---
            time_to_transcribe = transcription_end_time - recording_stop_time
            time_to_paste = pasting_end_time - recording_stop_time
            print("\n--- Statystyki Czasowe ---")
            print(f"⏱️ Czas do transkrypcji (od puszczenia klawisza): {time_to_transcribe:.2f}s")
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
        recording_stop_time = time.time() # ZAPISZ CZAS PUSZCZENIA KLAWISZA
        is_recording = False

# --- Główna Pętla Wykonawcza ---
if __name__ == "__main__":
    print("--- Uruchamianie Lokalnego Asystenta Dyktowania (Wersja Wsadowa) ---")
    app_settings = load_configuration()
    load_model(app_settings)
    hotkey_str = app_settings['hotkey']
    print(f"\n✅ Gotowy. Naciśnij i przytrzymaj '{hotkey_str}', aby nagrywać. Puść, aby transkrybować.")
    print("Naciśnij Ctrl+C, aby wyjść.")
    HOTKEY_COMBINATION = {keyboard.Key.ctrl, keyboard.Key.f8}
    current_keys = set()
    def on_press(key):
        if key in HOTKEY_COMBINATION:
            current_keys.add(key)
            if all(k in current_keys for k in HOTKEY_COMBINATION):
                start_recording_flag()
    def on_release(key):
        try:
            if key in HOTKEY_COMBINATION:
                stop_recording_flag()
                current_keys.clear()
        except KeyError:
            pass
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()