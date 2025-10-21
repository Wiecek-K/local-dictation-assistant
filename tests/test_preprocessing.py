# test_preprocessing.py
# Wersja 9: Finalna. Dodano końcowe podbicie głośności (+3dB).

import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import time

# --- Konfiguracja ---
INPUT_FILENAME = "sibilants_test.wav" 
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATE_BEFORE_SPECTROGRAM = False

# --- Finalne Parametry Potoku ---


SAMPLE_RATE = 16000             # [Hz] Częstotliwość próbkowania; standard dla modeli mowy.
DEESSER_THRESH_DB = -43         # [dB] Próg głośności, powyżej którego de-esser zaczyna tłumić syki.
DEESSER_ATTENUATION_DB = 13     # [dB] Siła, z jaką de-esser ścisza wykryte sybilanty.
DEESSER_FREQ_START = 6000       # [Hz] Dolna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_FREQ_END = 10000        # [Hz] Górna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_ATTACK_MS = 10           # [ms] Czas potrzebny na osiągnięcie pełnego tłumienia (wygładza początek).
DEESSER_RELEASE_MS = 30         # [ms] Czas powrotu do normalnej głośności (wygładza koniec, eliminuje trzaski).
FINAL_GAIN_DB = 6.0             # [dB] Końcowe podbicie głośności całego nagrania po przetworzeniu.

# --- Funkcje (bez zmian) ---
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

def create_spectrogram(audio_data, title, filename):
    fig, ax = plt.subplots(figsize=(15, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='linear', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Głośność (dB)')
    ax.set_ylim(0, 10000)
    ax.axhspan(DEESSER_FREQ_START, DEESSER_FREQ_END, color='g', alpha=0.2, label=f'De-esser Band')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Czas (s)"), ax.set_ylabel("Częstotliwość (Hz)"), ax.legend()
    plt.tight_layout(), plt.savefig(filename), plt.close()
    print(f"✅ Spektrogram zapisany w: {filename}")

# --- Główna Logika Skryptu ---
if __name__ == "__main__":
    print(f"--- Wczytywanie pliku audio: {INPUT_FILENAME} ---")
    try:
        audio_data_float32, sr = librosa.load(INPUT_FILENAME, sr=SAMPLE_RATE, mono=True)
        output_basename = os.path.splitext(os.path.basename(INPUT_FILENAME))[0]
        if GENERATE_BEFORE_SPECTROGRAM:
            create_spectrogram(audio_data_float32, f"Spektrogram - Oryginał ({INPUT_FILENAME})", os.path.join(OUTPUT_DIR, f"{output_basename}_spectrogram_BEFORE.png"))
    except Exception as e:
        print(f"❌ Błąd: {e}"), exit(1)

    print("\n--- Uruchamianie Finalnego Potoku Przetwarzania ---")
    audio_segment = AudioSegment((audio_data_float32 * 32767).astype(np.int16).tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    
    print("   - Krok 1: Normalizacja głośności...")
    normalized_segment = normalize(audio_segment)
    
    print(f"   - Krok 2: Aplikowanie de-essera z wygładzaniem...")
    deessed_segment = dynamic_de_esser_smooth(normalized_segment, DEESSER_THRESH_DB, DEESSER_FREQ_START, DEESSER_FREQ_END, DEESSER_ATTENUATION_DB, DEESSER_ATTACK_MS, DEESSER_RELEASE_MS)
    
    # NOWY KROK
    print(f"   - Krok 3: Podbicie głośności o +{FINAL_GAIN_DB} dB...")
    boosted_segment = deessed_segment + FINAL_GAIN_DB

    processed_before_nr = np.array(boosted_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
    
    print("   - Krok 4: Aplikowanie redukcji szumu...")
    final_audio_float32 = nr.reduce_noise(y=processed_before_nr, y_noise=processed_before_nr[:int(SAMPLE_RATE*0.5)], sr=SAMPLE_RATE, prop_decrease=0.85)

    print("\n--- Zapisywanie Wyników ---")
    output_filename_wav = os.path.join(OUTPUT_DIR, f"{output_basename}_processed.wav")
    write(output_filename_wav, SAMPLE_RATE, (final_audio_float32 * 32767).astype(np.int16))
    print(f"✅ Finalne audio zapisano w: {output_filename_wav}")
    
    create_spectrogram(final_audio_float32, f"Spektrogram - Po Przetworzeniu", os.path.join(OUTPUT_DIR, f"{output_basename}_spectrogram_AFTER.png"))
    print("\n--- Proces Zakończony ---")