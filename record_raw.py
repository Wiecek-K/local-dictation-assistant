# record_raw.py
# Proste narzędzie do nagrywania surowego, nieprzetworzonego audio do pliku WAV.
# Użycie: python record_raw.py <nazwa_pliku_wyjsciowego.wav>

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import argparse
import sys

# --- Konfiguracja ---
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # Mono

def main():
    # --- Parsowanie argumentów linii poleceń ---
    parser = argparse.ArgumentParser(
        description="Nagrywa surowe audio z mikrofonu do pliku WAV."
    )
    parser.add_argument(
        "filename",
        help="Ścieżka do pliku wyjściowego, np. 'sibilants_test.wav'"
    )
    args = parser.parse_args()

    # Lista do przechowywania nagranych fragmentów audio
    audio_frames = []

    def audio_callback(indata, frames, time, status):
        """Ta funkcja jest wywoływana dla każdego nowego bloku audio."""
        if status:
            print(f"Status strumienia: {status}", file=sys.stderr)
        audio_frames.append(indata.copy())

    print("--- Rozpoczynanie Nagrywania Surowego Audio ---")
    print(f"Plik wyjściowy: {args.filename}")
    print("\n🎙️  Mów teraz. Naciśnij Ctrl+C, aby zatrzymać nagrywanie.")

    try:
        # --- Uruchomienie strumienia nagrywania ---
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audio_callback):
            # Pętla czeka na przerwanie przez użytkownika (Ctrl+C)
            while True:
                sd.sleep(1000) # Czekaj w pętli, callback wykonuje pracę w tle

    except KeyboardInterrupt:
        print("\n🎙️  Nagrywanie zatrzymane przez użytkownika.")

    except Exception as e:
        print(f"❌ Wystąpił błąd podczas nagrywania: {e}")
        sys.exit(1)

    if not audio_frames:
        print("⚠️ Nie nagrano żadnego dźwięku. Plik nie został zapisany.")
        return

    print("\n💾 Przetwarzanie i zapisywanie pliku...")

    # Połącz wszystkie fragmenty w jeden ciągły sygnał
    audio_data_float32 = np.concatenate(audio_frames, axis=0)

    # Konwertuj format z float32 (-1.0 do 1.0) na int16 (standard dla WAV)
    audio_data_int16 = np.int16(audio_data_float32 * 32767)

    # Zapisz plik WAV
    try:
        write(args.filename, SAMPLE_RATE, audio_data_int16)
        print(f"✅ Plik z surowym audio został pomyślnie zapisany w: {args.filename}")
    except Exception as e:
        print(f"❌ Nie udało się zapisać pliku: {e}")

if __name__ == "__main__":
    main()