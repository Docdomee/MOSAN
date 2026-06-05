import pandas as pd
import streamlit as st
import ast
from ui_logger import log
from typing import List, Dict, Any

def load_csv_to_results(uploaded_file) -> list:
    """
    Carica un file CSV di risultati, lo analizza in modo robusto e lo converte
    nel formato corretto per st.session_state.results_log (una lista di dizionari).
    Gestisce in modo sicuro la colonna 'params' e garantisce la presenza delle colonne chiave.
    """
    log(f"Loading and parsing results from uploaded CSV file: {uploaded_file.name}")
    try:
        df = pd.read_csv(uploaded_file)

        # --- MODIFICA 1: GESTIONE SICURA DELLA COLONNA 'params' ---
        # La colonna 'params' nel CSV è una stringa. Dobbiamo riconvertirla in un dizionario.
        # ast.literal_eval è il modo più sicuro per farlo.
        if 'params' in df.columns:
            def safe_parse_params(x):
                # Controlla se il valore è una stringa non vuota
                if isinstance(x, str) and x.strip():
                    try:
                        # Tenta di interpretare la stringa come una struttura Python (es. un dizionario)
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        # Se fallisce, restituisce un dizionario vuoto per evitare errori
                        log(f"Warning: Could not parse params string, returning empty dict: {x}")
                        return {}
                # Se è già un dizionario (improbabile da CSV ma sicuro) o vuoto, lo restituisce
                return x if isinstance(x, dict) else {}

            df['params'] = df['params'].apply(safe_parse_params)
        
        # --- MODIFICA 2: CONTROLLO E AGGIUNTA DELLE COLONNE MANCANTI ---
        # Assicuriamoci che le colonne essenziali esistano. Se non esistono nel CSV, 
        # le creiamo con un valore di default per garantire l'integrità dei dati.
        required_cols = ['architecture', 'manifest_name', 'source']
        for col in required_cols:
            if col not in df.columns:
                df[col] = f"missing_{col}" # Aggiunge un valore di default
                log(f"Warning: Column '{col}' was missing in the uploaded CSV. Added with a default value.")

        # Converte il DataFrame pulito in una lista di dizionari, il formato atteso.
        parsed_results = df.to_dict('records')
        log(f"Successfully parsed {len(parsed_results)} records from CSV.")
        return parsed_results

    except Exception as e:
        log(f"ERROR: Failed to load or parse CSV file. Error: {e}")
        st.error(f"Error processing CSV file: {e}")
        return []

def update_best_result_from_log():
    """
    Scansiona il results_log corrente e aggiorna best_accuracy e best_params 
    nello stato della sessione.
    """
    results_log = st.session_state.get('results_log', [])
    if not results_log:
        return

    best_trial = max(results_log, key=lambda x: x.get('mean_accuracy', 0.0))
    current_best_accuracy = st.session_state.get('best_accuracy', 0.0)

    if best_trial.get('mean_accuracy', 0.0) > current_best_accuracy:
        log(f"Nuovo miglior risultato trovato nel file caricato: {best_trial['mean_accuracy']:.4f}")
        st.session_state.best_accuracy = best_trial['mean_accuracy']
        
        # Estrae solo i parametri, escludendo le metriche e la sorgente
        params_to_save = {k: v for k, v in best_trial.items() if k not in ['mean_accuracy', 'std_accuracy', 'source', 'experiment_name']}
        st.session_state.best_params = params_to_save

