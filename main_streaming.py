# FILE: main_streaming.py
# Wersja 4.0: Inteligentne cięcie fragmentów (RMS-VAD) z rozszerzonym logowaniem przyczyny cięcia.

import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import pyperclip
import subprocess
import logging

from pynput import keyboard, mouse

# Importy z refaktoryzowanych modułów
from src.audio_preprocessing import apply_preprocessing_pipeline, SAMPLE_RATE
from src.logger_setup import setup_loggers
from src.core_utils import load_configuration, load_model

# --- Inicjalizacja Loggerów ---
app_logger = logging.getLogger('app')
transcription_logger = logging.getLogger('transcription')
performance_logger = logging.getLogger('performance')

# --- Globalne Zmienne i Thread-safe State Management ---
audio_queue = queue.Queue()
is_recording = threading.Event() # Sygnalizuje, czy nagrywanie jest aktywne
full_transcript_context = ""
recording_start_time = 0
recording_stop_time = 0

# --- Funkcja Pomocnicza VAD (RMS-based) ---

def find_silence_split(audio_data, settings):
    """
    Analizuje bufor audio i szuka punktu cięcia opartego na ciszy (RMS).
    Zwraca indeks cięcia lub None.
    """
    # Parametry z config.ini
    RMS_THRESHOLD = settings['vad_rms_threshold']
    SILENCE_SAMPLES = int(settings['vad_silence_threshold_seconds'] * SAMPLE_RATE)
    MIN_CHUNK_SAMPLES = int(settings['vad_min_chunk_seconds'] * SAMPLE_RATE)
    
    # Oblicz RMS dla małych okien (np. 100ms)
    window_size = int(SAMPLE_RATE * 0.1)
    rms_values = np.array([
        np.sqrt(np.mean(audio_data[i:i + window_size]**2))
        for i in range(0, len(audio_data) - window_size, window_size)
    ])
    
    # Znajdź indeksy okien, które są poniżej progu ciszy
    silent_windows = np.where(rms_values < RMS_THRESHOLD)[0]
    
    if len(silent_windows) == 0:
        return None
    
    # Szukaj sekwencji ciszy o wymaganej długości
    required_windows = int(SILENCE_SAMPLES / window_size)
    
    # Iteruj od końca bufora, aby znaleźć ostatnią długą ciszę
    for i in range(len(silent_windows) - required_windows, -1, -1):
        # Sprawdź, czy mamy ciągłą sekwencję ciszy
        if np.all(np.diff(silent_windows[i : i + required_windows]) == 1):
            # Znaleziono sekwencję ciszy. Oblicz indeks cięcia.
            split_window_index = silent_windows[i]
            split_sample_index = split_window_index * window_size
            
            # Upewnij się, że fragment jest dłuższy niż minimalny czas
            if split_sample_index >= MIN_CHUNK_SAMPLES:
                return split_sample_index
                
    return None


# --- Producer-Consumer Logic ---

def recording_thread_func():
    """Wątek Producenta: Nagrywa audio i umieszcza je w kolejce."""
    app_logger.info("🎙️  Wątek nagrywający uruchomiony.")
    
    def audio_callback(indata, frames, time, status):
        """Callback wywoływany przez sounddevice."""
        if status:
            app_logger.warning(f"Status strumienia audio: {status}", file=sys.stderr)
        # Umieszcza fragment audio w kolejce
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
            while is_recording.is_set():
                time.sleep(0.1)
    except Exception as e:
        app_logger.error(f"❌ Błąd w wątku nagrywającym: {e}")
    
    app_logger.info("🎙️  Wątek nagrywający zakończony.")


def transcription_thread_func(settings, model_instance):
    """Wątek Konsumenta: Pobiera audio, przetwarza, transkrybuje i składa tekst."""
    global full_transcript_context
    full_transcript_context = ""
    
    transcription_logger.info("🧠 Wątek transkrybujący uruchomiony.")
    audio_buffer_list = []
    
    # Czas trwania bufora w próbkach
    MAX_BUFFER_SAMPLES = int(settings['vad_max_buffer_seconds'] * SAMPLE_RATE)
    MIN_CHUNK_SAMPLES = int(settings['vad_min_chunk_seconds'] * SAMPLE_RATE)
    
    # Pętla działa, dopóki nagrywanie jest aktywne LUB kolejka nie jest pusta
    while is_recording.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=0.01) 
            audio_buffer_list.append(audio_chunk)
            
            # Połącz bufor do analizy VAD
            current_buffer_data = np.concatenate(audio_buffer_list, axis=0).flatten().astype(np.float32)
            current_buffer_samples = len(current_buffer_data)
            
            split_index = None
            split_reason = None
            
            # 1. Sprawdź, czy można ciąć na ciszy (VAD)
            if current_buffer_samples >= MIN_CHUNK_SAMPLES:
                split_index = find_silence_split(current_buffer_data, settings)
                if split_index is not None:
                    split_reason = "VAD_SILENCE"
            
            # 2. Sprawdź, czy trzeba ciąć na sztywno (Max Buffer)
            if split_index is None and current_buffer_samples >= MAX_BUFFER_SAMPLES:
                split_index = MAX_BUFFER_SAMPLES
                split_reason = "MAX_BUFFER_LIMIT"
            
            # Jeśli znaleziono punkt cięcia
            if split_index is not None:
                # Podziel bufor na fragment do transkrypcji i resztę
                chunk_to_process = current_buffer_data[:split_index]
                remaining_data = current_buffer_data[split_index:]
                
                # Przetwórz i transkrybuj
                process_and_transcribe_chunk(chunk_to_process, settings, model_instance, is_final_chunk=False, split_reason=split_reason)
                
                # Zaktualizuj bufor: reszta danych staje się nowym buforem
                if len(remaining_data) > 0:
                    # Dodaj brakujący wymiar (kanał) do reszty danych
                    remaining_data = remaining_data.reshape(-1, 1) 
                    audio_buffer_list = [remaining_data]
                else:
                    audio_buffer_list = []
                
        except queue.Empty:
            if is_recording.is_set():
                continue
            else:
                break # Nagrywanie zatrzymane i kolejka pusta - zakończ
        except Exception as e:
            app_logger.error(f"❌ Błąd w wątku transkrybującym: {e}")
            break
            
    # POPRAWKA: Wymuś przetworzenie ostatniego, niepełnego bufora
    if audio_buffer_list:
        transcription_logger.info("🧠 Przetwarzanie ostatniego, niepełnego fragmentu...")
        raw_audio_data = np.concatenate(audio_buffer_list, axis=0).flatten().astype(np.float32)
        # NOWY ARGUMENT: split_reason
        process_and_transcribe_chunk(raw_audio_data, settings, model_instance, is_final_chunk=True, split_reason="END_OF_RECORDING")
            
    transcription_logger.info("🧠 Wątek transkrybujący zakończony.")


def process_and_transcribe_chunk(raw_audio_data, settings, model_instance, is_final_chunk, split_reason="END_OF_RECORDING"):
    """Przetwarza i transkrybuje pojedynczy fragment audio."""
    global full_transcript_context
    
    chunk_duration = len(raw_audio_data) / SAMPLE_RATE
    
    # NOWY LOG: Informacja o przyczynie cięcia
    reason_log = f" (Powód: {split_reason})"
    transcription_logger.info(f"🧠 Przetwarzanie fragmentu: {chunk_duration:.2f}s{reason_log}")
    
    # --- Krok 1: Preprocessing ---
    processed_audio = apply_preprocessing_pipeline(raw_audio_data)
    
    # --- Krok 2: Konfiguracja Transkrypcji ---
    lang_setting = settings['language']
    language_for_model = None if lang_setting.lower() == 'auto' else lang_setting
    
    # VAD jest teraz kontrolowany przez logikę cięcia, ale vad_filter w faster-whisper jest nadal użyteczny
    use_vad = settings['vad_filter'] 
    
    # Użycie kontekstu z poprzednich transkrypcji
    prompt = full_transcript_context.strip() if full_transcript_context.strip() else None
    
    # POPRAWKA OOM i POWTÓRZEŃ: Ograniczenie długości promptu do ostatnich 50 znaków
    MAX_PROMPT_LENGTH = 50 # ZMIANA: Zmniejszamy limit, aby zapobiec powtórzeniom
    if prompt and len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[-MAX_PROMPT_LENGTH:]
        transcription_logger.debug(f"   -> Ograniczono prompt do {MAX_PROMPT_LENGTH} znaków, aby zapobiec powtórzeniom.")
    
    transcription_start_time = time.time()
    
    # --- Krok 3: Transkrypcja ---
    segments_generator, info = model_instance.transcribe(
        processed_audio,
        language=language_for_model,
        beam_size=settings['beam_size'], 
        vad_filter=use_vad,
        log_prob_threshold=settings['log_prob_threshold'],
        no_speech_threshold=settings['no_speech_threshold'],
        initial_prompt=prompt,
        compression_ratio_threshold=2.4 # ZMIANA: Wymuszamy 2.4 (bardziej agresywny)
    )
    
    chunk_text = "".join(segment.text for segment in segments_generator).strip()
    transcription_duration = time.time() - transcription_start_time
    
    # --- Krok 4: Aktualizacja Kontekstu i Logowanie ---
    if chunk_text:
        # Dodajemy spację, aby oddzielić fragmenty
        full_transcript_context += chunk_text + " "
        transcription_logger.info(f"   -> Transkrybowany fragment: '{chunk_text}'")
        
        # Logowanie wydajności fragmentu
        rtf = float('inf')
        if chunk_duration > 0:
            rtf = transcription_duration / chunk_duration
        performance_logger.debug(f"   -> RTF fragmentu: {rtf:.3f} (Czas transkrypcji: {transcription_duration:.2f}s)")

# --- Hotkey Handling ---

rec_thread = None
trans_thread = None

def start_recording_flag(model_instance_arg, settings_arg):
    """Ustawia flagę nagrywania i uruchamia wątki."""
    global rec_thread, trans_thread, recording_start_time
    if is_recording.is_set():
        return
    
    app_logger.info("\n--- Skrót Aktywowany: Rozpoczynanie Nagrywania (Tryb Strumieniowy) ---")
    recording_start_time = time.time()
    is_recording.set() # Ustawia flagę
    
    # Wyczyść kolejkę na wszelki wypadek
    while not audio_queue.empty():
        audio_queue.get_nowait()

    rec_thread = threading.Thread(target=recording_thread_func)
    # Przekazujemy model i ustawienia do wątku Konsumenta
    trans_thread = threading.Thread(target=transcription_thread_func, args=(settings_arg, model_instance_arg))
    
    rec_thread.start()
    trans_thread.start()

def stop_recording_flag():
    """Czyści flagę nagrywania i czeka na zakończenie wątków."""
    global full_transcript_context, recording_stop_time
    if not is_recording.is_set():
        return
    
    app_logger.info("\n--- Skrót Zwolniony: Zatrzymywanie Nagrywania ---")
    recording_stop_time = time.time()
    is_recording.clear() # Zatrzymuje wątek Producenta
    
    transcription_finish_time = 0 
    
    # Czekaj na zakończenie wątku Konsumenta (opróżnienie kolejki i przetworzenie ostatniego bufora)
    if trans_thread:
        trans_thread.join()
        transcription_finish_time = time.time() 
    if rec_thread:
        rec_thread.join()
    
    # --- Finalizacja i Wklejanie ---
    final_text = full_transcript_context.strip()
    
    app_logger.info("\n--- Wynik Końcowy ---")
    app_logger.info(f"Tekst: {final_text}")

    if final_text:
        # Kopiowanie do schowka
        pyperclip.copy(final_text)
        app_logger.info("✅ Skopiowano do schowka.")
        
        # Wklejanie do aktywnego okna
        try:
            time.sleep(0.1)
            subprocess.run(["xdotool", "type", "--delay", "1", "--clearmodifiers", final_text], check=True)
            app_logger.info("✅ Wklejono do aktywnego okna.")
        except FileNotFoundError:
            app_logger.error("❌ BŁĄD: Polecenie 'xdotool' nie zostało znalezione.")
        except Exception as e:
            app_logger.error(f"❌ Błąd podczas wklejania tekstu: {e}")
            
    # Logowanie statystyk całkowitych
    total_duration = recording_stop_time - recording_start_time
    
    user_latency = transcription_finish_time - recording_stop_time if transcription_finish_time > 0 else 0
    
    performance_logger.info("\n--- Statystyki Czasowe (Całkowite) ---")
    performance_logger.info(f"⏱️ Czas nagrywania: {total_duration:.2f}s")
    performance_logger.info(f"⏱️ Latencja Użytkownika (od puszczenia klawisza do końca transkrypcji): {user_latency:.2f}s") 
    performance_logger.info(f"📝 Finalny tekst: {len(final_text)} znaków")
    
    app_logger.info("\n✅ Gotowy. Naciśnij i przytrzymaj skrót, aby nagrywać.")


# --- Hotkey Parsing (Skopiowane z main_simple.py) ---

def parse_hotkey(hotkey_string):
    hotkey_string = hotkey_string.lower().strip()
    if hotkey_string.startswith('mouse:'):
        button_name = hotkey_string.split(':')[1].strip()
        button_map = {
            'button4': mouse.Button.button8, 'button5': mouse.Button.button9,
            'left': mouse.Button.left, 'right': mouse.Button.right, 'middle': mouse.Button.middle,
        }
        if button_name in button_map:
            return {'type': 'mouse', 'button': button_map[button_name]}
        else:
            app_logger.warning(f"⚠️ OSTRZEŻENIE: Nieznany przycisk myszy w config.ini: '{button_name}'")
            return None
    else:
        keys = set()
        parts = hotkey_string.split('+')
        for part in parts:
            part = part.strip()
            key_name = part[1:-1] if part.startswith('<') and part.endswith('>') else part
            try:
                key = getattr(keyboard.Key, key_name)
                keys.add(key)
            except AttributeError:
                if len(key_name) == 1:
                    keys.add(keyboard.KeyCode.from_char(key_name))
                else:
                    app_logger.warning(f"⚠️ OSTRZEŻENIE: Nieznany klawisz w config.ini: '{key_name}'")
        return {'type': 'keyboard', 'key_set': keys}


# --- Main Execution ---

if __name__ == "__main__":
    setup_loggers()
    
    app_logger.info("--- Uruchamianie Lokalnego Asystenta Dyktowania (Wersja Strumieniowa) ---")
    
    # Wczytanie konfiguracji i modelu
    app_settings = load_configuration()
    model_instance = load_model(app_settings)
    
    hotkey_str = app_settings['hotkey']
    hotkey_config = parse_hotkey(hotkey_str)
    
    if not hotkey_config:
        app_logger.critical("❌ BŁĄD KRYTYCZNY: Nie udało się sparsować skrótu. Kończenie pracy.")
        sys.exit(1)

    app_logger.info(f"\n✅ Gotowy. Naciśnij i przytrzymaj '{hotkey_str}', aby nagrywać. Puść, aby transkrybować.")
    app_logger.info("Naciśnij Ctrl+C, aby wyjść.")
    
    listener = None
    if hotkey_config['type'] == 'keyboard':
        HOTKEY_COMBINATION = hotkey_config['key_set']
        current_keys = set()
        def on_press_keyboard(key):
            if key in HOTKEY_COMBINATION:
                current_keys.add(key)
                if current_keys == HOTKEY_COMBINATION: start_recording_flag(model_instance, app_settings)
        def on_release_keyboard(key):
            if key in HOTKEY_COMBINATION:
                stop_recording_flag()
                try: current_keys.remove(key)
                except KeyError: pass
        listener = keyboard.Listener(on_press=on_press_keyboard, on_release=on_release_keyboard)
    elif hotkey_config['type'] == 'mouse':
        MOUSE_BUTTON = hotkey_config['button']
        def on_click_mouse(x, y, button, pressed):
            if button == MOUSE_BUTTON:
                if pressed: start_recording_flag(model_instance, app_settings)
                else: stop_recording_flag()
        listener = mouse.Listener(on_click=on_click_mouse)

    if listener:
        with listener:
            listener.join()