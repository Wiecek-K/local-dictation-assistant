# main_streaming.py

import configparser
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- Global Variables & Thread-safe State Management ---
model = None
audio_queue = queue.Queue()
is_recording = threading.Event()
transcribed_texts = []

# --- Configuration Loading (unchanged) ---
def load_configuration():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        if not config.sections():
            raise FileNotFoundError("config.ini is empty or not found.")
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='small'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='int8'),
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

# --- NEW: Producer-Consumer Logic ---

def recording_thread_func(settings):
    """
    Producer thread: Captures audio from the microphone and puts it into the queue.
    """
    SAMPLE_RATE = 16000
    
    def audio_callback(indata, frames, time, status):
        """This function is called by sounddevice for each new audio chunk."""
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)
        # The audio data is a numpy array, which we put directly into the queue.
        audio_queue.put(indata.copy())

    print("🎙️  Recording thread started. Speak into your microphone.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
            while is_recording.is_set():
                time.sleep(0.1) # The callback is doing the work, this loop just keeps the context alive.
    except Exception as e:
        print(f"❌ Error in recording thread: {e}")
    print("🎙️  Recording thread finished.")


def transcription_thread_func(settings):
    """
    Consumer thread: Processes audio from the queue and transcribes it.
    """
    global transcribed_texts
    transcribed_texts = []
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = settings['chunk_seconds']
    CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE

    print("🧠 Transcription thread started. Waiting for audio...")
    
    while is_recording.is_set() or not audio_queue.empty():
        try:
            # Accumulate audio chunks from the queue to form a larger segment for transcription.
            audio_buffer = []
            current_samples = 0
            while current_samples < CHUNK_SAMPLES:
                try:
                    audio_chunk = audio_queue.get(timeout=1) # Wait for audio for up to 1 second.
                    audio_buffer.append(audio_chunk)
                    current_samples += len(audio_chunk)
                    if not is_recording.is_set():
                        break # Stop accumulating if recording has stopped.
                except queue.Empty:
                    # If the queue is empty, break the inner loop.
                    break
            
            if not audio_buffer:
                if not is_recording.is_set():
                    break # Exit if recording is done and queue is empty.
                continue # Continue waiting for audio.

            # Process the accumulated audio data.
            audio_data = np.concatenate(audio_buffer, axis=0).flatten()
            
            print(f"🧠 Processing a chunk of {len(audio_data)/SAMPLE_RATE:.2f} seconds...")
            segments, _ = model.transcribe(
                audio_data,
                language=settings['language'],
                beam_size=5,
                vad_filter=settings['vad_filter']
            )
            
            chunk_text = "".join(segment.text for segment in segments).strip()
            if chunk_text:
                print(f"  -> Transcribed chunk: '{chunk_text}'")
                transcribed_texts.append(chunk_text)

        except queue.Empty:
            if not is_recording.is_set():
                break
            continue
        except Exception as e:
            print(f"❌ Error in transcription thread: {e}")
            break
            
    print("🧠 Transcription thread finished.")


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Local Dictation Assistant (POC) ---")
    app_settings = load_configuration()
    load_model(app_settings)
    
    print("\n--- Starting Live Test ---")
    print("The system will now record and transcribe for 15 seconds.")
    print("Please start speaking now...")

    # Set the recording event to start the threads
    is_recording.set()

    # Start the producer and consumer threads
    rec_thread = threading.Thread(target=recording_thread_func, args=(app_settings,))
    trans_thread = threading.Thread(target=transcription_thread_func, args=(app_settings,))
    
    rec_thread.start()
    trans_thread.start()

    # Let it run for 15 seconds
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    # Signal the threads to stop
    print("\n--- Stopping Live Test ---")
    is_recording.clear()

    # Wait for the threads to finish their work
    rec_thread.join()
    trans_thread.join()

    # Combine and print the final result
    final_text = " ".join(transcribed_texts)
    print("\n--- Final Result ---")
    print(f"Full transcribed text: {final_text}")
    
    print("\n--- Program shutting down. ---")