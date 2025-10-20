# main_streaming.py
# Advanced version with real-time streaming transcription.

import configparser
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import pyperclip
import subprocess
from faster_whisper import WhisperModel
from pynput import keyboard

# --- Global Variables & Thread-safe State Management ---
model = None
audio_queue = queue.Queue()
is_recording = threading.Event()
full_transcript_context = ""

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
            'vad_filter': config.getboolean('settings', 'vad_filter', fallback=True),
            'chunk_seconds': config.getint('streaming', 'chunk_seconds', fallback=5)
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

# --- Producer-Consumer Logic ---

def recording_thread_func(settings):
    SAMPLE_RATE = 16000
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print("🎙️  Recording thread started.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
            while is_recording.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"❌ Error in recording thread: {e}")
    print("🎙️  Recording thread finished.")


def transcription_thread_func(settings):
    global full_transcript_context
    full_transcript_context = ""
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = settings['chunk_seconds']
    
    print("🧠 Transcription thread started.")
    audio_buffer = []
    while is_recording.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_buffer.append(audio_chunk)
            buffer_duration = len(np.concatenate(audio_buffer)) / SAMPLE_RATE
            if buffer_duration >= CHUNK_SECONDS:
                process_audio_chunk(np.concatenate(audio_buffer, axis=0).flatten(), settings)
                audio_buffer = []
        except queue.Empty:
            if is_recording.is_set():
                continue
            elif audio_buffer:
                process_audio_chunk(np.concatenate(audio_buffer, axis=0).flatten(), settings)
                audio_buffer = []
            else:
                break
        except Exception as e:
            print(f"❌ Error in transcription thread: {e}")
            break
    print("🧠 Transcription thread finished.")

def process_audio_chunk(audio_data, settings):
    global full_transcript_context
    SAMPLE_RATE = 16000
    MIN_CHUNK_DURATION = 0.5

    if len(audio_data) / SAMPLE_RATE < MIN_CHUNK_DURATION:
        return

    print(f"🧠 Processing a chunk of {len(audio_data)/SAMPLE_RATE:.2f} seconds...")

    prompt = full_transcript_context if full_transcript_context else "To jest test polskiej transkrypcji."

    segments, _ = model.transcribe(
        audio_data,
        language=settings['language'],
        beam_size=5,
        vad_filter=settings['vad_filter'],
        initial_prompt=prompt
    )
    
    chunk_text = "".join(segment.text for segment in segments).strip()
    if chunk_text:
        print(f"  -> Transcribed chunk: '{chunk_text}'")
        full_transcript_context += chunk_text + " "

# --- Hotkey Handling ---

rec_thread = None
trans_thread = None
listener = None

def start_recording(settings):
    global rec_thread, trans_thread
    if is_recording.is_set():
        return
    
    print("\n--- Hotkey Activated: Starting Recording ---")
    is_recording.set()
    while not audio_queue.empty():
        audio_queue.get_nowait()

    rec_thread = threading.Thread(target=recording_thread_func, args=(settings,))
    trans_thread = threading.Thread(target=transcription_thread_func, args=(settings,))
    rec_thread.start()
    trans_thread.start()

def stop_recording():
    global full_transcript_context, listener
    if not is_recording.is_set():
        return
    
    print("\n--- Hotkey Released: Stopping Recording ---")
    is_recording.clear()
    
    if rec_thread:
        rec_thread.join()
    if trans_thread:
        trans_thread.join()
    
    final_text = full_transcript_context.strip()
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
    
    print("\n--- Task complete. Stopping listener. ---")
    if listener:
        listener.stop()

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Local Dictation Assistant (Streaming Version) ---")
    app_settings = load_configuration()
    load_model(app_settings)
    
    hotkey_str = app_settings['hotkey']
    print(f"\n✅ Ready. Press and hold '{hotkey_str}' to record. Press Ctrl+C to exit.")
    
    HOTKEY_COMBINATION = {
        keyboard.Key.ctrl,
        keyboard.Key.f8
    }
    current_keys = set()
    hotkey_activated = False

    def on_press(key):
        global hotkey_activated
        if key in HOTKEY_COMBINATION:
            current_keys.add(key)
            if all(k in current_keys for k in HOTKEY_COMBINATION) and not hotkey_activated:
                hotkey_activated = True
                start_recording(app_settings)

    def on_release(key):
        global hotkey_activated
        try:
            current_keys.remove(key)
            if key in HOTKEY_COMBINATION and hotkey_activated:
                hotkey_activated = False
                stop_recording()
        except KeyError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()
    
    print("--- Program finished. ---")