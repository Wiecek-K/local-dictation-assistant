# Dokumentacja Problemów: Przechwytywanie Audio w Systemie Linux

Data: 2025-10-23
System: Pop!\_OS 22.04 LTS
Serwer Dźwięku: PipeWire

## Podsumowanie

Podczas fazy POC napotkano na krytyczny, trudny do zdiagnozowania problem, który objawiał się **zawieszaniem się aplikacji podczas próby nagrywania dźwięku**. Problem występował niezależnie od użytej biblioteki Pythona (`sounddevice`, `PyAudio`, `soundcard`) oraz narzędzi systemowych (`arecord`, `ffmpeg`).

Ostatecznie zdiagnozowano, że przyczyną nie był błąd w kodzie, lecz **niestabilny stan systemowego serwera dźwięku (PipeWire)**, który "zgubił" dostęp do urządzenia wejściowego.

---

### Objawy Problemu

- Aplikacja (lub narzędzie wiersza poleceń) zawieszała się bezterminowo po próbie otwarcia strumienia nagrywania.
- Brak było komunikatów o błędach, z wyjątkiem `Timeout` na bardzo niskim poziomie sterownika ALSA.
- Programy nie reagowały na sygnał `Ctrl+C` w sposób poprawny.
- Problem pojawił się po "Generalnym Remoncie" systemu, czyli reinstalacji sterowników NVIDIA i komponentów CUDA.

### Proces Diagnostyczny

1.  **Izolacja Komponentów:** Stworzono serię minimalistycznych skryptów testowych, aby wyizolować źródło problemu. Testy wykazały, że `pynput` i `threading` działają poprawnie, a problem pojawia się przy każdej interakcji z systemem audio.
2.  **Zmiana Bibliotek:** Próby zmiany biblioteki z `sounddevice` na `PyAudio`, `soundcard`, a nawet na zewnętrzne procesy `arecord` i `ffmpeg` kończyły się tym samym błędem (`Timeout` lub zawieszeniem), co wskazywało na problem systemowy.
3.  **Diagnostyka Systemowa:** Użycie narzędzia `pw-top` pokazało, że serwer dźwięku PipeWire tworzy połączenia, ale **nie przesyła danych audio** z mikrofonu do aplikacji.
4.  **Przełom:** Ostatecznym rozwiązaniem okazał się **twardy reset środowiska audio**: restart komputera, fizyczne odłączenie i ponowne podłączenie mikrofonu USB oraz zmiana domyślnego urządzenia wejściowego w ustawieniach systemowych. Po tych czynnościach system "odzyskał" dostęp do mikrofonu.

### Wnioski i Procedura Postępowania na Przyszłość

**Główny wniosek:** Problemy z przechwytywaniem audio w Linuksie są często symptomem problemu na poziomie systemu/sterownika, a nie błędu w kodzie aplikacji.

**Procedura Diagnostyczna (Checklista) w Przypadku Problemów z Nagrywaniem:**

W przypadku ponownego wystąpienia problemów z zawieszaniem się aplikacji podczas nagrywania, należy postępować według poniższej checklisty **przed rozpoczęciem debugowania kodu**:

1.  **Weryfikacja w Ustawieniach Systemowych (Krok 1):**

    - Otwórz `Ustawienia > Dźwięk > Wejście`.
    - Upewnij się, że wybrany jest właściwy mikrofon.
    - **Mów do mikrofonu i sprawdź, czy pasek "Poziom wejściowy" reaguje.** Jeśli pasek jest martwy, problem jest na 100% systemowy.

2.  **Test w Zewnętrznej Aplikacji (Krok 2):**

    - Użyj niezawodnej, zewnętrznej aplikacji, aby zweryfikować działanie mikrofonu.
    - **Rekomendacja: Narzędzie "Głos i Wideo" w ustawieniach Discorda.**
    - Kliknij przycisk "Sprawdźmy" i mów do mikrofonu. Jeśli Discord Cię "słyszy" (pasek się zapala), oznacza to, że podstawowa warstwa audio działa.

3.  **Twardy Reset Usług Audio (Krok 3):**

    - Jeśli powyższe testy zawiodą, wykonaj twardy reset serwera dźwięku:
      ```bash
      systemctl --user restart pipewire pipewire-pulse wireplumber
      ```
    - Po wykonaniu komendy, wróć do Kroku 1 i sprawdź, czy mikrofon zaczął reagować.

4.  **Reset Fizyczny (Krok 4):**
    - Odłącz i podłącz ponownie urządzenie USB (mikrofon).
    - W ostateczności, zrestartuj komputer.

Dopiero po potwierdzeniu, że mikrofon działa na poziomie systemu (pasek w ustawieniach reaguje, Discord go słyszy), można przystąpić do debugowania kodu aplikacji.
