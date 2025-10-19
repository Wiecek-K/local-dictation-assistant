# main_streaming.py

import configparser
import sys
import time
from faster_whisper import WhisperModel

# --- Global Variables ---
# We will store the loaded model in a global variable
model = None

# --- Configuration Loading ---

def load_configuration():
    """
    Loads settings from config.ini and returns them as a dictionary.
    Provides sensible defaults if the file or a setting is missing.
    """
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        if not config.sections():
            raise FileNotFoundError("config.ini is empty or not found.")
            
        settings = {
            'model_path': config.get('settings', 'model_path', fallback='small'),
            'device': config.get('settings', 'device', fallback='cuda'),
            'compute_type': config.get('settings', 'compute_type', fallback='float16'),
            'hotkey': config.get('settings', 'hotkey', fallback='<ctrl>+<shift>+f1'),
            'language': config.get('settings', 'language', fallback='pl'),
            'vad_filter': config.getboolean('settings', 'vad_filter', fallback=True),
            'chunk_seconds': config.getint('streaming', 'chunk_seconds', fallback=5)
        }
        print("Configuration loaded successfully.")
        return settings
    except (FileNotFoundError, configparser.Error) as e:
        print(f"Error loading config.ini: {e}")
        print("Please ensure config.ini exists and is correctly formatted.")
        sys.exit(1)

# --- Core Functions ---

def load_model(settings):
    """
    Loads the Whisper model into memory based on the provided settings.
    This is the most resource-intensive part of the startup.
    """
    global model
    model_path = settings['model_path']
    device = settings['device']
    compute_type = settings['compute_type']

    print("\n--- Model Loading ---")
    print(f"Attempting to load model: '{model_path}'")
    print(f"Device: '{device}', Compute Type: '{compute_type}'")
    
    # This can take a while, especially on the first run when the model is downloaded.
    start_time = time.time()
    
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        end_time = time.time()
        print(f"✅ Model loaded successfully in {end_time - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"❌ Critical Error: Failed to load the Whisper model.")
        print(f"   Error details: {e}")
        print("\n--- Troubleshooting ---")
        print("1. Ensure you have a compatible Nvidia driver installed.")
        print("2. Verify that CUDA Toolkit and cuDNN are correctly installed and configured.")
        print("3. Check if the model name in config.ini is correct.")
        print("4. Make sure you have enough VRAM available on your GPU.")
        sys.exit(1)

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Local Dictation Assistant (POC) ---")
    
    # Step 1: Load configuration from the file
    app_settings = load_configuration()
    
    # Step 2: Load the AI model
    load_model(app_settings)
    
    print("\n--- Verification ---")
    print("The model is now loaded into your GPU's VRAM.")
    print("Open a new terminal and run the command 'nvidia-smi' to verify.")
    print("You should see a python process consuming VRAM.")
    
    # We keep the script running to allow for verification
    try:
        input("\nPress Enter to exit the program...")
    except KeyboardInterrupt:
        pass
    
    print("--- Program shutting down. ---")