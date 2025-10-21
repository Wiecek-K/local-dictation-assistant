# Dokumentacja Systemu Logowania

Ten dokument wyjaśnia, jak działa segmentowy system logowania w projekcie. Pozwala on na precyzyjną kontrolę nad tym, jakie informacje są wyświetlane w konsoli, co jest kluczowe podczas testowania i normalnego użytkowania.

Cała konfiguracja odbywa się w pliku `config.ini`, w sekcji `[logging]`.

## 1. Poziomy Logowania

System używa standardowych poziomów logowania. Ustawienie danego poziomu (np. `INFO`) powoduje wyświetlanie komunikatów z tego poziomu oraz **wszystkich poziomów o wyższym priorytecie**.

Hierarchia poziomów (od najniższego do najwyższego priorytetu):

1.  **`DEBUG`**: Najbardziej szczegółowy poziom. Pokazuje absolutnie wszystko. Idealny do głębokiego debugowania konkretnego modułu.
2.  **`INFO`**: Standardowy poziom. Pokazuje kluczowe informacje o przebiegu działania aplikacji (np. "Nagrywanie rozpoczęte", "Model załadowany").
3.  **`WARNING`**: Poziom dla ostrzeżeń. Informuje o niekrytycznych problemach, które nie przerywają działania aplikacji, ale mogą być niepożądane (np. "Nieznany klawisz w konfiguracji").
4.  **`ERROR`**: Poziom dla błędów. Pokazuje tylko błędy, które mogły przerwać daną operację (np. "Błąd podczas wklejania tekstu").

## 2. Kategorie (Loggery)

Logi zostały podzielone na cztery niezależne kategorie. Każdą z nich można kontrolować osobno w pliku `config.ini`.

| Logger              | Klucz Konfiguracyjny      | Domyślny Poziom | Przeznaczenie                                                           |
| :------------------ | :------------------------ | :-------------- | :---------------------------------------------------------------------- |
| **`app`**           | `log_level_app`           | `INFO`          | Główne komunikaty o stanie aplikacji, widoczne dla użytkownika.         |
| **`preprocessing`** | `log_level_preprocessing` | `WARNING`       | Szczegółowe informacje o każdym kroku potoku przetwarzania audio.       |
| **`transcription`** | `log_level_transcription` | `INFO`          | Informacje związane bezpośrednio z modelem Whisper (np. wykryty język). |
| **`performance`**   | `log_level_performance`   | `INFO`          | Statystyki czasowe i metryki wydajności po zakończeniu transkrypcji.    |

---

## 3. Przykładowe Scenariusze Użycia

Poniżej znajdują się przykłady, jak dostosować logi do konkretnych potrzeb.

### Scenariusz 1: Normalne Użytkowanie (Konfiguracja Domyślna)

W tym trybie logi z potoku `preprocessing` są ukryte, aby nie zaśmiecać konsoli. Widzisz tylko kluczowe informacje.

**Konfiguracja `config.ini`:**

```ini
[logging]
log_level_app = INFO
log_level_preprocessing = WARNING
log_level_transcription = INFO
log_level_performance = INFO
```

**Przykładowy wygląd logów:**

```
--- Uruchamianie Lokalnego Asystenta Dyktowania ---
Konfiguracja załadowana pomyślnie.
✅ Model załadowany pomyślnie w 8.98s.

✅ Gotowy. Naciśnij i przytrzymaj 'mouse:button5', aby nagrywać.

🎙️  Nagrywanie... Mów teraz.
🎙️  Nagrywanie zatrzymane.
🔊 Uruchamianie potoku przetwarzania wstępnego audio...
🔊 Przetwarzanie wstępne zakończone pomyślnie (całkowity czas: 0.71s).
🧠 Rozpoczynanie transkrypcji...
   -> Długość audio (po preprocessingu): 12.23s
   -> Wykryto język: pl (prawdopodobieństwo: 1.00)

--- Wynik Końcowy ---
Tekst: To jest przykładowa transkrypcja.
✅ Skopiowano do schowka.
✅ Wklejono do aktywnego okna.

--- Statystyki Czasowe ---
🎧 Długość nagrania: 12.23s
🧠 Czas samej transkrypcji (z iteracją): 1.74s
...

```

### Scenariusz 2: Debugowanie Potoku Audio

Chcesz zobaczyć, ile czasu zajmuje każdy krok w `preprocessing`. Zmieniasz **tylko jedną linię**.

**Konfiguracja `config.ini`:**

```ini
[logging]
log_level_app = INFO
log_level_preprocessing = DEBUG  # <-- ZMIANA TUTAJ
log_level_transcription = INFO
log_level_performance = INFO
```

**Przykładowy wygląd logów (pojawią się nowe linie):**

```
...
🎙️  Nagrywanie zatrzymane.
🔊 Uruchamianie potoku przetwarzania wstępnego audio...
   - Krok 1: Normalizacja głośności...
     (czas: 0.00s)
   - Krok 2: Aplikowanie de-essera z wygładzaniem...
     (czas: 0.60s)
   - Krok 3: Podbicie głośności o +6.0 dB...
     (czas: 0.00s)
   - Krok 4: Aplikowanie redukcji szumu...
     (czas: 0.11s)
🔊 Przetwarzanie wstępne zakończone pomyślnie (całkowity czas: 0.71s).
🧠 Rozpoczynanie transkrypcji...
...
```

### Scenariusz 3: Tryb "Cichy" (Tylko Wynik)

Chcesz widzieć jak najmniej logów, praktycznie tylko finalny tekst.

**Konfiguracja `config.ini`:**

```ini
[logging]
log_level_app = WARNING
log_level_preprocessing = WARNING
log_level_transcription = WARNING
log_level_performance = WARNING
```

**Przykładowy wygląd logów:**

```
--- Wynik Końcowy ---
Tekst: To jest przykładowa transkrypcja.
```

_(W tym trybie nadal będą widoczne ewentualne błędy i ostrzeżenia)._```
