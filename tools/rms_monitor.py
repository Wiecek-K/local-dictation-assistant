# FILE: tools/rms_monitor.py
# Narzędzie do pomiaru RMS (Root Mean Square - Głośności) w czasie rzeczywistym.

import sounddevice as sd
import numpy as np
import sys
import time

# --- Konfiguracja ---
SAMPLE_RATE = 16000  # Częstotliwość próbkowania (musi być zgodna z projektem)
CHANNELS = 1         # Mono
BLOCK_SIZE = 1024    # Rozmiar bloku do przetwarzania (wpływa na responsywność)

def calculate_rms(data):
    """Oblicza RMS (Root Mean Square) dla bloku danych audio."""
    # Używamy np.float32, jak w projekcie
    return np.sqrt(np.mean(data**2))

def audio_callback(indata, frames, time, status):
    """Callback wywoływany dla każdego bloku audio."""
    if status:
        sys.stderr.write(f"Status strumienia: {status}\n")
    
    # Oblicz RMS
    rms = calculate_rms(indata)
    
    # Wyczyść linię i wyświetl wynik
    sys.stdout.write(f"\rRMS: {rms:.5f} | Mów teraz, aby zobaczyć pik. Milcz, aby zobaczyć szum tła.")
    sys.stdout.flush()

def main():
    print("--- Monitor RMS (Głośności) w Czasie Rzeczywistym ---")
    print(f"Częstotliwość próbkowania: {SAMPLE_RATE} Hz")
    print("Naciśnij Ctrl+C, aby zakończyć.")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype='float32', 
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            # Pętla czeka na przerwanie przez użytkownika (Ctrl+C)
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMonitorowanie zakończone przez użytkownika.")
    except Exception as e:
        print(f"\n❌ Wystąpił błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()