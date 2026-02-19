import sqlite3
import pandas as pd
import numpy as np
import pymaid
import time
import json
import os
import multiprocessing
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
NUM_WORKERS = 4
BATCH_SIZE = 100

# INPUT: The master DB with all predictions
DB_PREDICTIONS = "/media/samia/DATA/mounts/gpu2/synapse_detection/predictions/octo_cns/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db"

# OLD LOG: Just for safety, we check this to ensure no duplicates
DB_OLD_LOG = "/media/samia/DATA/PhD/codebases/pymaid/push_results_60k_70k_round3.sqlite"

# NEW OUTPUT: We will save the new progress here
DB_NEW_LOG = "push_results_70k_plus.sqlite"
TABLE_NAME = "push_log"

# CATMAID Credentials
CATMAID_URL = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/#"
CATMAID_TOKEN = "4fe0c0572331265c61bf6ba4acd77cbed9792276"
CATMAID_AUTH_NAME = "SYLee"
CATMAID_AUTH_PASS = "Blue Skies"
PROJECT_ID = 32

# --- HELPER FUNCTIONS ---

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
    if x is None: return None
    try: return int(x)
    except: return None

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

def process_batch(chunk, rm):
    rows_out = []
    links = []
    link_row_idxs = []

    for r in chunk.itertuples(index=True):
        current_idx = r.Index 
        
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

        rows_out.append({
            "idx": current_idx,
            "post_id": post_id,
            "pre_id": pre_id,
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
        })
        
        if treenode_id is not None and connector_id is not None:
            links.append((treenode_id, connector_id, "postsynaptic_to"))
            link_row_idxs.append(len(rows_out) - 1)

    if links:
        link_resp_raw, link_exc = retry_call(pymaid.link_connector, links)

        if link_exc is None and isinstance(link_resp_raw, list):
            for j, item in enumerate(link_resp_raw):
                row_i = link_row_idxs[j] if j < len(link_row_idxs) else None
                if row_i is not None:
                    ok = isinstance(item, dict) and item.get("message") == "success"
                    rows_out[row_i]["linked_ok"] = 1 if ok else 0
                    if not ok:
                        rows_out[row_i]["link_error"] = str(item)
        else:
            err = link_exc or str(link_resp_raw)
            for row_i in link_row_idxs:
                rows_out[row_i]["linked_ok"] = 0
                rows_out[row_i]["link_error"] = err

    return rows_out

def worker_task(worker_id, df_subset):
    """Worker process."""
    db_path = f"progress_worker_{worker_id}.sqlite"
    conn = sqlite3.connect(db_path)
    init_db(conn, TABLE_NAME)
    
    try:
        rm = pymaid.CatmaidInstance(CATMAID_URL, CATMAID_TOKEN, CATMAID_AUTH_NAME, CATMAID_AUTH_PASS, PROJECT_ID)
    except Exception as e:
        print(f"[Worker {worker_id}] Connection failed: {e}")
        return

    pbar = tqdm(total=len(df_subset), position=worker_id, desc=f"Worker {worker_id}")
    
    for start_idx in range(0, len(df_subset), BATCH_SIZE):
        batch = df_subset.iloc[start_idx : start_idx + BATCH_SIZE]
        try:
            results = process_batch(batch, rm)
            if results:
                pd.DataFrame(results).to_sql(TABLE_NAME, conn, if_exists="append", index=False)
                conn.commit()
        except Exception as e:
            with open(f"worker_{worker_id}_error.log", "a") as f:
                f.write(f"Batch start {start_idx} failed: {e}\n")
        
        pbar.update(len(batch))

    pbar.close()
    conn.close()

# --- MAIN CONTROLLER ---

def get_processed_coordinates(db_list):
    """
    Returns a set of (pre_x, pre_y, pre_z, post_x, post_y, post_z) 
    that have already been processed in any of the provided DBs.
    """
    processed = set()
    print("Scanning logs to exclude duplicates...")
    
    for db_path in db_list:
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
            if cursor.fetchone():
                query = f"SELECT pre_x, pre_y, pre_z, post_x, post_y, post_z FROM {TABLE_NAME}"
                df_done = pd.read_sql_query(query, conn)
                for row in df_done.itertuples(index=False):
                    processed.add(tuple(row))
            conn.close()
        except Exception as e:
            print(f"Warning reading {db_path}: {e}")
            
    print(f"Found {len(processed)} previously pushed synapses (to be excluded).")
    return processed

def main():
    # 1. Load ALL Predictions
    print(f"Loading ALL predictions from {DB_PREDICTIONS}...")
    if not os.path.exists(DB_PREDICTIONS):
        print("❌ Error: DB_PREDICTIONS file not found!")
        return

    conn = sqlite3.connect(DB_PREDICTIONS)
    query = """
    SELECT 
        pre.id AS pre_id, pre.score, pre.z AS pre_z, pre.y AS pre_y, pre.x AS pre_x,
        post.id AS post_id, post.z AS post_z, post.y AS post_y, post.x AS post_x
    FROM pre_sites pre
    JOIN pre_post_mapping map ON pre.id = map.pre_id
    JOIN post_sites post ON post.id = map.post_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Total predictions in DB: {len(df)}")
    
    # 2. FILTER: Only take Z >= 70,000
    print("Filtering for Z >= 70,000 (The rest of the data)...")
    df = df[df["pre_z"] >= 70000].copy()
    print(f"Rows remaining after Z filter: {len(df)}")

    # 3. SAFETY: Exclude anything found in old logs or current worker logs
    # We check the old 60-70k log just in case there is boundary overlap
    # We also check 'push_results_70k_plus.sqlite' and worker logs to allow RESUMING this new run
    db_files = [DB_OLD_LOG, DB_NEW_LOG] + [f"progress_worker_{i}.sqlite" for i in range(NUM_WORKERS)]
    processed_coords = get_processed_coordinates(db_files)

    if processed_coords:
        print("Filtering out already processed synapses...")
        df['coord_key'] = list(zip(df.pre_x, df.pre_y, df.pre_z, df.post_x, df.post_y, df.post_z))
        df = df[~df['coord_key'].isin(processed_coords)].copy()
        df.drop(columns=['coord_key'], inplace=True)
    
    if len(df) == 0:
        print("✅ Nothing left to process! Everything > 70k is done.")
        return

    print(f"Final Count to Push: {len(df)} synapses.")

    # 4. Sort by Z for efficient spatial processing
    df.sort_values(by=['pre_z', 'pre_y', 'pre_x'], inplace=True)

    # 5. Launch Workers
    chunks = np.array_split(df, NUM_WORKERS)
    processes = []
    
    print(f"Launching {NUM_WORKERS} workers...")
    for i in range(NUM_WORKERS):
        if len(chunks[i]) > 0:
            p = multiprocessing.Process(target=worker_task, args=(i, chunks[i]))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
        
    print("All workers finished.")
    
    # 6. Merge Worker DBs into the NEW log file
    print(f"Merging worker logs into {DB_NEW_LOG}...")
    conn_new = sqlite3.connect(DB_NEW_LOG)
    init_db(conn_new, TABLE_NAME)
    
    for i in range(NUM_WORKERS):
        w_db = f"progress_worker_{i}.sqlite"
        if os.path.exists(w_db):
            try:
                # Attach and merge
                conn_new.execute(f"ATTACH DATABASE '{w_db}' AS w{i}")
                conn_new.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM w{i}.{TABLE_NAME}")
                conn_new.commit()
                conn_new.execute(f"DETACH DATABASE w{i}")
            except Exception as e:
                print(f"Error merging worker {i}: {e}")
    conn_new.close()
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
