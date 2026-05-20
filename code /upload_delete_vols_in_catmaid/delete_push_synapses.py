import os
import sys
import sqlite3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import pandas as pd
import pymaid
from tqdm import tqdm

# --- CATMAID Credentials ---
CATMAID_URL = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/#"
CATMAID_TOKEN = "c"
CATMAID_AUTH_NAME = "x"
CATMAID_AUTH_PASS = "X"
PROJECT_ID = 32

# --- Configuration ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 10))

# Centralized queue for logging failures safely across threads in this process
failure_queue = queue.Queue()

def setup_failure_db(failures_db_path):
    """Initializes the failure logging database specific to this master process."""
    conn = sqlite3.connect(failures_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS failed_deletions (
            id TEXT,
            entity_type TEXT,
            source_db TEXT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_failure_worker(failures_db_path):
    """Background thread that writes queued failures to the specific DB."""
    conn = sqlite3.connect(failures_db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    while True:
        item = failure_queue.get()
        if item is None:
            break # Stop signal
            
        entity_id, entity_type, source_db, error_msg = item
        cursor.execute(
            "INSERT INTO failed_deletions (id, entity_type, source_db, error_message) VALUES (?, ?, ?, ?)",
            (str(entity_id), str(entity_type), source_db, str(error_msg))
        )
        conn.commit()
        failure_queue.task_done()
        
    conn.close()

def delete_entity(entity_id, entity_type, source_db):
    """
    Uses pymaid to handle the deletion of a node or connector.
    """
    try:
        # pymaid natively handles the API routing for nodes vs connectors
        pymaid.delete_nodes(entity_id, entity_type, no_prompt=True, remote_instance=None)
        return True

    except Exception as e:
        error_msg = str(e)
        # Check if the exception indicates the node is already gone
        if "404" in error_msg or "does not exist" in error_msg.lower():
            failure_queue.put((entity_id, entity_type, source_db, "Skipped: Not Found (Already deleted or missing)"))
        else:
            failure_queue.put((entity_id, entity_type, source_db, f"Failed: {error_msg}"))
        return False

def extract_tasks_from_db(db_path):
    """
    Extracts treenode_ids and connector_ids from the push_log using pandas
    and queues them as individual deletion tasks.
    """
    tasks = []
    try:
        conn = sqlite3.connect(db_path)
        df_db = pd.read_sql_query("SELECT * FROM push_log", conn)
        conn.close()
        
        if df_db.empty:
            return tasks

        for index, row in df_db.iterrows():
            # Extract and queue treenode_id
            if pd.notna(row.get('treenode_id')):
                tasks.append((int(row['treenode_id']), "NODE"))
            
            # Extract and queue connector_id
            if pd.notna(row.get('connector_id')):
                tasks.append((int(row['connector_id']), "CONNECTOR"))
            
    except Exception as e:
        print(f"\n[{db_path}] Error reading push_log table: {e}")
        
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Concurrent deletion of CATMAID nodes/connectors using pymaid.")
    parser.add_argument("db_path", help="Path to the SQLite database containing the push_log.")
    args = parser.parse_args()

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        sys.exit(1)

    # Automatically derive a unique failures database name based on the input database
    db_dir = os.path.dirname(db_path)
    db_basename = os.path.basename(db_path)
    db_name_without_ext = os.path.splitext(db_basename)[0]
    failures_db_path = os.path.join(db_dir, f"{db_name_without_ext}_failures.sqlite")

    # Initialize the global pymaid instance using provided credentials
    try:
        rm = pymaid.CatmaidInstance(CATMAID_URL, CATMAID_TOKEN, CATMAID_AUTH_NAME, CATMAID_AUTH_PASS, PROJECT_ID)
    except Exception as e:
        print(f"Failed to initialize pymaid instance: {e}")
        sys.exit(1)

    setup_failure_db(failures_db_path)

    # Start the failure logger thread passing the unique failure db path
    logger_thread = threading.Thread(target=log_failure_worker, args=(failures_db_path,))
    logger_thread.start()

    print(f"Reading {db_path}...")
    tasks = extract_tasks_from_db(db_path)
    
    if not tasks:
        print(f"[{db_path}] No valid IDs found or error reading table. Exiting.")
        failure_queue.put(None)
        logger_thread.join()
        sys.exit(0)

    print(f"Queued {len(tasks)} tasks from {db_basename}. Starting deletion with {MAX_WORKERS} workers...\n")

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(delete_entity, t[0], t[1], db_path): t for t in tasks
        }
        
        # Wrap as_completed with tqdm for the progress bar
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Deleting IDs ({db_basename})"):
            try:
                if future.result():
                    success_count += 1
            except Exception as exc:
                task = future_to_task[future]
                failure_queue.put((task[0], task[1], db_path, f"Exception: {str(exc)}"))

    print(f"\n[{db_basename}] Finished. Successfully deleted {success_count}/{len(tasks)} entities.")
    print(f"[{db_basename}] Failures logged to: {failures_db_path}")

    # Shutdown logger safely
    failure_queue.put(None)
    logger_thread.join()

if __name__ == "__main__":
    main()
