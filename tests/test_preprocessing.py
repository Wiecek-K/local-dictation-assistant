# FILE: tests/test_preprocessing.py
# Wersja 11: Zrefaktoryzowana do użycia zoptymalizowanego potoku i parsowania argumentów linii poleceń.

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import sys
import argparse # NOWY IMPORT

# --- Konfiguracja Ścieżek i Importów ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    # Importujemy zoptymalizowany potok i parametry
    from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE, DEESSER_FREQ_START, DEESSER_FREQ_END
except ImportError as e:
    print(f"❌ BŁĄD KRYTYCZNY: Nie można zaimportować modułu: {e}")
    sys.exit(1)

# --- Konfiguracja ---
OUTPUT_DIR = os.path.join(PARENT_DIR, "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATE_BEFORE_SPECTROGRAM = True 

# --- Funkcje Pomocnicze ---

def create_spectrogram(audio_data, title, filename):
    """Generuje i zapisuje spektrogram."""
    fig, ax = plt.subplots(figsize=(15, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='linear', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Głośność (dB)')
    ax.set_ylim(0, 10000)
    
    # Używamy zaimportowanych parametrów dla wizualizacji pasma de-essera
    ax.axhspan(DEESSER_FREQ_START, DEESSER_FREQ_END, color='g', alpha=0.2, label=f'De-esser Band ({DEESSER_FREQ_START}-{DEESSER_FREQ_END} Hz)')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Czas (s)"), ax.set_ylabel("Częstotliwość (Hz)"), ax.legend()
    plt.tight_layout(), plt.savefig(filename), plt.close()
    print(f"✅ Spektrogram zapisany w: {filename}")

# --- Główna Logika Skryptu ---
if __name__ == "__main__":
    # --- Parsowanie argumentów linii poleceń ---
    parser = argparse.ArgumentParser(
        description="Przetwarza plik audio za pomocą zoptymalizowanego potoku preprocessingu."
    )
    parser.add_argument(
        "filepath",
        help="Ścieżka do pliku audio do przetworzenia (np. tests/sibilants_test.wav)"
    )
    parser.add_argument(
        "--no-spectrogram",
        action="store_true",
        help="Wyłącza generowanie spektrogramów."
    )
    args = parser.parse_args()
    input_path = args.filepath
    
    print(f"--- Wczytywanie pliku audio: {os.path.basename(input_path)} ---")
    
    if not os.path.exists(input_path):
        print(f"❌ BŁĄD: Brak pliku wejściowego: {input_path}.")
        sys.exit(1)
        
    try:
        # Wczytujemy surowe dane
        audio_data_float32, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        output_basename = os.path.splitext(os.path.basename(input_path))[0]
        
        if GENERATE_BEFORE_SPECTROGRAM and not args.no_spectrogram:
            create_spectrogram(audio_data_float32, f"Spektrogram - Oryginał ({output_basename})", os.path.join(OUTPUT_DIR, f"{output_basename}_spectrogram_BEFORE.png"))
            
    except Exception as e:
        print(f"❌ Błąd wczytywania audio: {e}"), exit(1)

    print("\n--- Uruchamianie ZOPTYMALIZOWANEGO Potoku Przetwarzania ---")
    
    # KLUCZOWA ZMIANA: Używamy zoptymalizowanej funkcji z modułu src
    final_audio_float32 = apply_preprocessing_pipeline(audio_data_float32.copy())

    print("\n--- Zapisywanie Wyników ---")
    output_filename_wav = os.path.join(OUTPUT_DIR, f"{output_basename}_processed_optimized.wav")
    
    # Konwersja z float32 na int16 do zapisu WAV
    audio_data_int16 = np.int16(final_audio_float32 * 32767)
    
    try:
        write(output_filename_wav, SAMPLE_RATE, audio_data_int16)
        print(f"✅ Finalne audio zapisano w: {output_filename_wav}")
    except Exception as e:
        print(f"❌ Nie udało się zapisać pliku: {e}")
        
    if not args.no_spectrogram:
        create_spectrogram(final_audio_float32, f"Spektrogram - Po Zoptymalizowanym Przetworzeniu", os.path.join(OUTPUT_DIR, f"{output_basename}_spectrogram_AFTER_OPTIMIZED.png"))
    print("\n--- Proces Zakończony ---")