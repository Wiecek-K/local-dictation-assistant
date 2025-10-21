# audio_processing.py
"""
Moduł odpowiedzialny za zaawansowane przetwarzanie wstępne sygnału audio.
"""
import time
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

# --- Globalne Parametry Potoku Przetwarzania ---
SAMPLE_RATE = 16000                 # [Hz] Częstotliwość próbkowania; standard dla modeli mowy.
DEESSER_THRESH_DB = -43             # [dB] Próg głośności, powyżej którego de-esser zaczyna tłumić sybilanty.
DEESSER_ATTENUATION_DB = 13         # [dB] Siła, z jaką de-esser ścisza wykryte sybilanty.
DEESSER_FREQ_START = 6000           # [Hz] Dolna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_FREQ_END = 10000            # [Hz] Górna granica pasma częstotliwości, w którym działa de-esser.
DEESSER_ATTACK_MS = 10              # [ms] Czas potrzebny na osiągnięcie pełnego tłumienia (wygładza początek).
DEESSER_RELEASE_MS = 30             # [ms] Czas powrotu do normalnej głośności (wygładza koniec, eliminuje trzaski).
FINAL_GAIN_DB = 6.0                 # [dB] Końcowe podbicie głośności całego nagrania po przetworzeniu.

def dynamic_de_esser_smooth(audio_segment, threshold_db, freq_start, freq_end, attenuation_db, attack_ms, release_ms):
    """
    Dynamiczny de-esser z wygładzaniem (fade in/out) w celu eliminacji trzasków.
    Działa na fragmentach (chunks) i stosuje tłumienie tylko wtedy, gdy jest to konieczne.
    """
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
    """
    Aplikuje pełny potok przetwarzania wstępnego na surowych danych audio.
    :param audio_data_float32: Dane audio jako numpy array w formacie float32.
    :return: Przetworzone dane audio jako numpy array w formacie float32.
    """
    print("🔊 Uruchamianie potoku przetwarzania wstępnego audio...")
    pipeline_start_time = time.time()
    last_step_time = pipeline_start_time
    try:
        # Konwersja numpy array (float32) na pydub AudioSegment (int16)
        audio_data_int16 = np.int16(audio_data_float32 * 32767)
        audio_segment = AudioSegment(
            audio_data_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=audio_data_int16.dtype.itemsize,
            channels=1
        )

        # Krok 1: Normalizacja
        print(f"   - Krok 1: Normalizacja głośności...")
        normalized_segment = normalize(audio_segment)
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time

        # Krok 2: De-esser
        print(f"   - Krok 2: Aplikowanie de-essera z wygładzaniem...")
        deessed_segment = dynamic_de_esser_smooth(
            normalized_segment,
            DEESSER_THRESH_DB, DEESSER_FREQ_START, DEESSER_FREQ_END,
            DEESSER_ATTENUATION_DB, DEESSER_ATTACK_MS, DEESSER_RELEASE_MS
        )
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time

        # Krok 3: Podbicie głośności
        print(f"   - Krok 3: Podbicie głośności o +{FINAL_GAIN_DB} dB...")
        boosted_segment = deessed_segment + FINAL_GAIN_DB
        current_time = time.time()
        print(f"     (czas: {current_time - last_step_time:.2f}s)")
        last_step_time = current_time

        # Konwersja z powrotem do numpy array (float32) dla noisereduce
        processed_before_nr = np.array(boosted_segment.get_array_of_samples(), dtype=np.float32) / 32767.0

        # Krok 4: Redukcja szumu
        print("   - Krok 4: Aplikowanie redukcji szumu...")
        # Użyj pierwszych 0.5s jako próbki szumu
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