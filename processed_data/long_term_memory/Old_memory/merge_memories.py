# merge_memories.py
import json
import os
import uuid

# --- CONFIGURAZIONE ---
# 1. Inserisci qui i percorsi ai tuoi due file di memoria
FILE_1_PATH = "/app/processed_data/long_term_memory/global_agent_memory_001.json"
FILE_2_PATH = "/app/processed_data/long_term_memory/global_agent_memory_002.json"

# 2. Questo sarà il nuovo file di memoria unificato e corretto
# Assicurati che corrisponda al percorso usato dal tuo programma principale
OUTPUT_FILE_PATH = "/app/processed_data/long_term_memory/global_agent_memory.json"
# --------------------


def migrate_memories():
    """
    Carica uno o più file di memoria, li unisce, rimuove i duplicati
    e assicura che ogni entry abbia un 'finding_id' univoco.
    """
    all_memories = []

    # Carica le memorie da tutti i file specificati
    for file_path in [FILE_1_PATH, FILE_2_PATH]:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    if content:
                        all_memories.extend(json.loads(content))
                print(f"✅ Caricate {len(json.loads(content))} memorie da: {file_path}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Errore nel leggere il file {file_path}: {e}. File saltato.")
        else:
            print(f"ℹ️ File non trovato, saltato: {file_path}")

    if not all_memories:
        print("❌ Nessuna memoria da processare. Uscita.")
        return

    # Rimuovi i duplicati basandoti sul testo della scoperta ('finding')
    unique_findings = set()
    merged_memories = []
    for memory in all_memories:
        # Assicurati che la memoria sia un dizionario e contenga 'finding'
        if isinstance(memory, dict) and "finding" in memory:
            if memory["finding"] not in unique_findings:
                unique_findings.add(memory["finding"])
                merged_memories.append(memory)

    print(f"👍 Trovate {len(merged_memories)} memorie uniche dopo la fusione.")

    # Aggiungi un ID univoco a ogni memoria che non ne ha uno
    updated_count = 0
    final_memories = []
    for memory in merged_memories:
        # Controlla se 'finding_id' è assente, nullo o una stringa vuota
        if not memory.get("finding_id"):
            memory["finding_id"] = str(uuid.uuid4())
            updated_count += 1
        final_memories.append(memory)

    if updated_count > 0:
        print(f"🔧 Aggiornate {updated_count} memorie con un nuovo ID univoco.")

    # Salva il nuovo file di memoria unificato
    try:
        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(final_memories, f, indent=4)
        print(f"🎉 Processo completato! Il nuovo file di memoria unificato è stato salvato in: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio del file finale: {e}")


if __name__ == "__main__":
    migrate_memories()
