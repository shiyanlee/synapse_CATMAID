"""
Author: Samia Mohinta
Affiliation: Cambridge, UK
"""

import sqlite3
import pandas as pd
import numpy as np
import pymaid
import time
import json
import os
import multiprocessing
from tqdm import tqdm

# --- CONFIGURATION ---
NUM_WORKERS = 4  # Adjust based on your CPU
BATCH_SIZE = 100  # Number of rows to process before batch-linking and committing
DB_PREDICTIONS = "/media/samia/DATA/mounts/gpu2/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db"
DB_PROGRESS_MAIN = "/media/samia/DATA/PhD/codebases/pymaid/push_results_60k_70k_round3.sqlite/push_results_60k_70k_round3.sqlite"  # Main history DB
TABLE_NAME = "push_log"

# CATMAID Credentials
CATMAID_URL = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/#"
CATMAID_TOKEN = "x"
CATMAID_AUTH_NAME = "x"
CATMAID_AUTH_PASS = "x"
PROJECT_ID = 32

# --- HELPER FUNCTIONS (From Notebook) ---

def retry_call(fn, *args, retries=3, sleep_base=1.0, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))
    return None, repr(last_err)

def resp_error(resp):
    if isinstance(resp, dict) and resp.get("error"):
        return resp.get("error")
    return None

def normalize_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def first_dict_from_response(resp):
    if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
        return resp[0]
    if isinstance(resp, dict):
        return resp
    return {}

def init_db(conn, table_name):
    """Creates the log table if it doesn't exist."""
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      idx INTEGER,
      post_id INTEGER,
      pre_id INTEGER, 
      
      pre_x REAL, pre_y REAL, pre_z REAL,
      post_x REAL, post_y REAL, post_z REAL,

      treenode_id INTEGER,
      skeleton_id INTEGER,
      edition_time TEXT,
      node_error TEXT,

      connector_id INTEGER,
      connector_edition_time TEXT,
      created_links TEXT,
      connector_error TEXT,

      linked_ok INTEGER,
      link_error TEXT,
      
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

# --- WORKER LOGIC ---

def process_chunk_logic(chunk, rm):
    """
    Core logic extracted from the notebook. 
    Processes a dataframe chunk: creates nodes, connectors, and links them.
    """
    rows_out = []
    links = []
    link_row_idxs = []

    for r in chunk.itertuples(index=False):
        # We assume unique ID is preserved in the column 'original_idx' or we use pre/post IDs
        # Using pre_id/post_id as primary reference
        pre_id = normalize_int(r.pre_id)
        post_id = normalize_int(r.post_id)
        
        pre_x, pre_y, pre_z = float(r.pre_x), float(r.pre_y), float(r.pre_z)
        post_x, post_y, post_z = float(r.post_x), float(r.post_y), float(r.post_z)

        # 1) Add Node (Post-synaptic)
        node_resp_raw, node_exc = retry_call(
            pymaid.add_node,
            (post_x, post_y, post_z),
            parent_id=None,
            radius=-1,
            confidence=5,
        )
        node_resp = first_dict_from_response(node_resp_raw)
        node_err = node_exc or resp_error(node_resp)

        treenode_id = normalize_int(node_resp.get("treenode_id") or node_resp.get("id")) if node_err is None else None
        skeleton_id  = normalize_int(node_resp.get("skeleton_id")) if node_err is None else None
        edition_time = node_resp.get("edition_time") if node_err is None else None

        # 2) Add Connector (Pre-synaptic)
        connector_id = None
        connector_edition_time = None
        created_links = None
        conn_err = None

        if treenode_id is not None:
            conn_resp_raw, conn_exc = retry_call(
                pymaid.add_connector,
                (pre_x, pre_y, pre_z),
                remote_instance=None
            )
            conn_resp = first_dict_from_response(conn_resp_raw)
            conn_err = conn_exc or resp_error(conn_resp)

            if conn_err is None:
                connector_id = normalize_int(conn_resp.get("connector_id") or conn_resp.get("id") or conn_resp.get("new_connector_id"))
                connector_edition_time = conn_resp.get("connector_edition_time")
                created_links = conn_resp.get("created_links")

                if connector_id is None:
                    conn_err = f"Connector created but connector_id missing. Keys={list(conn_resp.keys())}"

        # Prepare log row
        row_data = {
            "post_id": post_id,
            "pre_id": pre_id, # Added pre_id for resume logic
            "pre_x": pre_x, "pre_y": pre_y, "pre_z": pre_z,
            "post_x": post_x, "post_y": post_y, "post_z": post_z,
            "treenode_id": treenode_id,
            "skeleton_id": skeleton_id,
            "connector_id": connector_id,
            "edition_time": edition_time,
            "node_error": node_err,
            "connector_error": conn_err,
            "connector_edition_time": connector_edition_time,
            "created_links": str(created_links) if created_links else None,
            "linked_ok": 0,
            "link_error": None,
        }
        
        # 3) Prepare Link
        if treenode_id is not None and connector_id is not None:
            links.append((treenode_id, connector_id, "postsynaptic_to"))
            link_row_idxs.append(len(rows_out)) # Store index of this row in rows_out
        
        rows_out.append(row_data)

    # Batch Link Call
    if links:
        link_resp_raw, link_exc = retry_call(pymaid.link_connector, links)

        # Process link responses
        if link_exc is None and isinstance(link_resp_raw, list):
            for j, item in enumerate(link_resp_raw):
                row_i = link_row_idxs[j] if j < len(link_row_idxs) else None
                if row_i is not None:
                    ok = isinstance(item, dict) and item.get("message") == "success"
                    rows_out[row_i]["linked_ok"] = 1 if ok else 0
                    if not ok:
                        rows_out[row_i]["link_error"] = str(item)
        else:
            # Batch failure
            err = link_exc or str(link_resp_raw)
            for row_i in link_row_idxs:
                rows_out[row_i]["linked_ok"] = 0
                rows_out[row_i]["link_error"] = err

    return rows_out

def worker_task(worker_id, df_subset):
    """
    Main worker process loop.
    """
    # 1. Setup Worker DB
    db_path = f"progress_worker_{worker_id}.sqlite"
    conn = sqlite3.connect(db_path)
    init_db(conn, TABLE_NAME)
    
    # 2. Setup Pymaid (Must be inside process)
    try:
        rm = pymaid.CatmaidInstance(CATMAID_URL, CATMAID_TOKEN, CATMAID_AUTH_NAME, CATMAID_AUTH_PASS, PROJECT_ID)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to connect to CATMAID: {e}")
        return

    # 3. Process in batches
    # tqdm position allows multiple progress bars to stack neatly
    pbar = tqdm(total=len(df_subset), position=worker_id, desc=f"Worker {worker_id}")
    
    # Iterate in mini-batches to allow frequent commits and partial progress
    for start_idx in range(0, len(df_subset), BATCH_SIZE):
        batch = df_subset.iloc[start_idx : start_idx + BATCH_SIZE]
        
        try:
            results = process_chunk_logic(batch, rm)
            
            # Save results
            if results:
                pd.DataFrame(results).to_sql(TABLE_NAME, conn, if_exists="append", index=False)
                conn.commit()
                
        except Exception as e:
            # Catch-all to prevent worker death on df errors, log to a text file
            with open(f"worker_{worker_id}_error.log", "a") as f:
                f.write(f"Batch {start_idx} failed: {e}\n")
        
        pbar.update(len(batch))

    pbar.close()
    conn.close()

# --- MAIN ORCHESTRATION ---

def get_processed_pairs():
    """
    Scans the main progress DB and all worker DBs to find
    (pre_id, post_id) pairs that are already done.
    """
    processed = set()
    
    db_files = [DB_PROGRESS_MAIN] + [f"progress_worker_{i}.sqlite" for i in range(NUM_WORKERS)]
    
    print("Scanning progress databases for existing records...")
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                c = sqlite3.connect(db_file)
                # Check if table exists
                cursor = c.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
                if cursor.fetchone():
                    # Read pre_id, post_id. NOTE: Ensure your notebook logic saves these columns.
                    # If your old DB only has 'idx', we might need to rely on that or handle differently.
                    # The updated worker logic above saves pre_id/post_id explicitly.
                    try:
                        # Attempt to read pre_id/post_id
                        data = pd.read_sql_query(f"SELECT pre_id, post_id FROM {TABLE_NAME}", c)
                        for _, row in data.iterrows():
                            processed.add((row['pre_id'], row['post_id']))
                    except Exception:
                        # Fallback for old schema if it lacks pre_id column?
                        # Assuming we are starting fresh or compatible schema.
                        print(f"Warning: Could not read pre_id/post_id from {db_file}. Check schema.")
                c.close()
            except Exception as e:
                print(f"Error reading {db_file}: {e}")
                
    return processed

def main():
    # 1. Load Source Data
    print(f"Loading predictions from {DB_PREDICTIONS}...")
    conn = sqlite3.connect(DB_PREDICTIONS)
    query = """
    SELECT 
        pre.id AS pre_id,
        pre.score,
        pre.z AS pre_z, pre.y AS pre_y, pre.x AS pre_x,
        post.id AS post_id,
        post.z AS post_z, post.y AS post_y, post.x AS post_x
    FROM pre_sites pre
    JOIN pre_post_mapping map ON pre.id = map.pre_id
    JOIN post_sites post ON post.id = map.post_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Total rows loaded: {len(df)}")

    # 2. Filter Already Processed
    processed_pairs = get_processed_pairs()
    if processed_pairs:
        print(f"Found {len(processed_pairs)} processed pairs. Filtering...")
        # Create a tuple key for filtering
        # We use a temporary index to speed this up
        df['temp_key'] = list(zip(df.pre_id, df.post_id))
        df = df[~df['temp_key'].isin(processed_pairs)].copy()
        df.drop(columns=['temp_key'], inplace=True)
        print(f"Rows remaining: {len(df)}")
    else:
        print("No previous progress found (or schema mismatch). Processing all.")

    if len(df) == 0:
        print("All data processed! Exiting.")
        return

    # 3. Spatial Partitioning
    # Sort by Z (pre_z) to ensure workers stay in specific spatial regions
    print("Sorting by Z for spatial partitioning...")
    df.sort_values(by=['pre_z', 'pre_y', 'pre_x'], inplace=True)

    # 4. Split Data
    chunks = np.array_split(df, NUM_WORKERS)
    
    # 5. Launch Workers
    print(f"Launching {NUM_WORKERS} workers...")
    processes = []
    
    for i in range(NUM_WORKERS):
        if len(chunks[i]) > 0:
            p = multiprocessing.Process(target=worker_task, args=(i, chunks[i]))
            processes.append(p)
            p.start()
        else:
            print(f"Worker {i} has no data assigned.")

    for p in processes:
        p.join()
        
    print("All workers finished.")

if __name__ == "__main__":
    # Required for Windows/macOS multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
