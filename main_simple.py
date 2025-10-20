# main_simple.py
# A simplified version for quality verification.
# Records a full audio clip, saves it to a file, and transcribes it in one go.

import os
from datetime import datetime
import configparser
import sys
import time
import numpy as np
import sounddevice as sd
import pyperclip
import subprocess
import threading
from faster_whisper import WhisperModel
from pynput import keyboard
from scipy.io.wavfile import write

# --- Global Variables ---
model = None
is_recording = False
audio_frames = []
app_settings = {} # To store settings globally for the hotkey thread

# --- Configuration Loading ---
def load_configuration():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='small'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
            'hotkey': config.get('settings', 'hotkey', fallback='<ctrl>+f8'),
            'language': config.get('settings', 'language', fallback='pl'),
        }
        print("Configuration loaded successfully.")
        return settings
    except (FileNotFoundError, configparser.Error) as e:
        print(f"Error loading config.ini: {e}")
        sys.exit(1)

# --- Core Functions ---
def load_model(settings):
    global model
    model_path = settings['model_path']
    device = settings['device']
    compute_type = settings['compute_type']
    print("\n--- Model Loading ---")
    print(f"Attempting to load model: '{model_path}'")
    print(f"Device: '{device}', Compute Type: '{compute_type}'")
    start_time = time.time()
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        end_time = time.time()
        print(f"✅ Model loaded successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load the Whisper model: {e}")
        sys.exit(1)

def record_and_transcribe(settings):
    global audio_frames
    SAMPLE_RATE = 16000
    
    print("🎙️  Recording... Speak now.")
    
    audio_frames = []
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)
        audio_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    
    while is_recording:
        time.sleep(0.1)
        
    stream.stop()
    stream.close()
    print("🎙️  Recording stopped.")

    if not audio_frames:
        print("No audio recorded.")
        return

    audio_data = np.concatenate(audio_frames, axis=0).flatten().astype(np.float32)

    # --- Save to File Logic ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate filenames for both wav and txt
    wav_filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
    txt_filename = os.path.join(output_dir, f"recording_{timestamp}.txt")

    try:
        audio_data_int16 = np.int16(audio_data * 32767)
        write(wav_filename, SAMPLE_RATE, audio_data_int16)
        print(f"✅ Audio saved to {wav_filename}")
    except Exception as e:
        print(f"❌ Error saving audio file: {e}")

    # --- Transcription Logic ---
    print("🧠 Transcribing... (this may take a moment)")
    start_time = time.time()
    
    segments, _ = model.transcribe(
        audio_data,
        language=settings['language'],
        beam_size=5
    )
    
    end_time = time.time()
    
    final_text = "".join(segment.text for segment in segments).strip()
    
    print(f"🧠 Transcription finished in {end_time - start_time:.2f} seconds.")
    
    # --- NEW: Save transcript to .txt file ---
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"✅ Transcript saved to {txt_filename}")
    except Exception as e:
        print(f"❌ Error saving transcript file: {e}")

    print("\n--- Final Result ---")
    print(f"Full transcribed text: {final_text}")

    if final_text:
        pyperclip.copy(final_text)
        print("✅ Copied to clipboard.")
        try:
            time.sleep(0.1)
            subprocess.run(["xdotool", "type", "--clearmodifiers", final_text], check=True)
            print("✅ Pasted into active window.")
        except Exception as e:
            print(f"❌ Error during pasting: {e}")

def start_recording_flag():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_and_transcribe, args=(app_settings,)).start()

def stop_recording_flag():
    global is_recording
    if is_recording:
        is_recording = False

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Local Dictation Assistant (Simple Quality Test) ---")
    app_settings = load_configuration()
    load_model(app_settings)
    
    hotkey_str = app_settings['hotkey']
    print(f"\n✅ Ready. Press and hold '{hotkey_str}' to record. Release to transcribe.")
    print("Press Ctrl+C to exit.")
    
    HOTKEY_COMBINATION = {
        keyboard.Key.ctrl,
        keyboard.Key.f8
    }
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