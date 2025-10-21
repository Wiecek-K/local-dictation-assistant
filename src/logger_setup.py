# src/logger_setup.py
"""
Moduł do centralnej, precyzyjnej konfiguracji loggerów aplikacji.
"""
import logging
import configparser
import os
import sys

def setup_loggers():
    """
    Wczytuje konfigurację z config.ini i ustawia poziomy oraz handlery
    tylko dla zdefiniowanych loggerów aplikacji, unikając globalnej konfiguracji.
    """
    config = configparser.ConfigParser()
    # Ścieżka do config.ini jest względna do głównego katalogu projektu
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
    config.read(config_path)

    # Definicja loggerów i ich kluczy w pliku konfiguracyjnym
    loggers_config = {
        'app': 'log_level_app',
        'preprocessing': 'log_level_preprocessing',
        'transcription': 'log_level_transcription',
        'performance': 'log_level_performance'
    }

    # Utwórz jeden handler i formatter, aby ponownie ich używać
    log_format = '%(message)s'
    formatter = logging.Formatter(log_format)
    # Użyj sys.stdout, aby logi zachowywały się jak standardowy print
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    for name, config_key in loggers_config.items():
        logger = logging.getLogger(name)
        
        # Pobierz poziom z config.ini, domyślnie INFO
        level_str = config.get('logging', config_key, fallback='INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)
        
        # Kluczowa zmiana: dodajemy handler tylko do naszych loggerów
        # i tylko wtedy, gdy jeszcze go nie mają.
        if not logger.handlers:
            logger.addHandler(handler)
        
        # Zapobiegaj propagacji do roota, aby uniknąć podwójnych logów
        # i konfliktów z innymi bibliotekami.
        logger.propagate = False