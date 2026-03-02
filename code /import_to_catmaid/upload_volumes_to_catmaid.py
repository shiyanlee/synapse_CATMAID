import os
import pymaid
import trimesh

def sync_pk_volumes(folder_path: str, rm: pymaid.CatmaidInstance):
    """
    Overwrites existing 'PK' volumes in CATMAID with .stl files from a local folder.
    
    Args:
        folder_path (str): Path to the folder containing your .stl files.
        rm (pymaid.CatmaidInstance): Authenticated pymaid CATMAID connection object.
    """
    print("Fetching existing volumes from CATMAID...")
    
    # 1. Fetch and parse existing volumes to identify 'PK' volumes
    vols_endpoint = f"{rm.project_id}/volumes/"
    try:
        existing_vols = rm.fetch(vols_endpoint)
        
        pk_vol_ids = []
        
        # Handle variations in CATMAID API return structures (dict vs. list)
        if isinstance(existing_vols, dict):
            if "volumes" in existing_vols:
                existing_vols = existing_vols["volumes"]
            else:
                for k, v in existing_vols.items():
                    if isinstance(v, dict) and v.get("name", "").startswith("PK"):
                        pk_vol_ids.append(v.get("id", k))
                        
        if isinstance(existing_vols, list):
            for v in existing_vols:
                if isinstance(v, dict) and v.get("name", "").startswith("PK"):
                    pk_vol_ids.append(v.get("id"))
                elif isinstance(v, list) and len(v) > 1:
                    # In cases where the API returns a nested list: [[id, "volume_name", ...], ...]
                    if str(v[1]).startswith("PK"):
                        pk_vol_ids.append(v[0])

        # 2. Delete the identified 'PK' volumes using a DELETE request
        for vol_id in pk_vol_ids:
            if vol_id is not None:
                print(f"Deleting existing volume ID: {vol_id}")
                rm.fetch(f"{rm.project_id}/volumes/{vol_id}/", method="DELETE")
                
    except Exception as e:
        print(f"Warning: Could not fetch or delete existing volumes. Details: {e}")

    # 3. Locate and upload new PK_ .stl files
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path is not a valid directory: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.startswith("PK_") and filename.lower().endswith(".stl"):
            file_path = os.path.join(folder_path, filename)
            
            # Use the filename without the .stl extension as the volume name
            vol_name = os.path.splitext(filename)[0] 
            
            print(f"\nLoading {filename}...")
            try:
                # Load the .stl into a mesh object
                mesh = trimesh.load(file_path)
                
                # Extract faces and vertices into the dictionary format pymaid expects
                mesh_dict = {
                    'vertices': mesh.vertices,
                    'faces': mesh.faces
                }
                
                print(f"Uploading {vol_name}...")
                pymaid.upload_volume(mesh_dict, name=vol_name, remote_instance=rm)
                print(f"Successfully uploaded: {vol_name}")
                
            except Exception as e:
                print(f"Failed to process or upload {filename}: {e}")

# ==========================================
# Execution Setup
# ==========================================
if __name__ == "__main__":
    # Configure your credentials and path here
    TARGET_FOLDER = "/media/samia/DATA/mounts/cephfs/nblast_mclayton/octo_alingnedtoSeymour/all_stl_files"
    
    OCTO_URL     = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/"   # no trailing slash
    OCTO_TOKEN   = "90f00590555eb256f256d1062e38521cbe180293"
    OCTO_PROJECT = 32      # project ID for Octo
    
    # Initialize the pymaid instance
    rm = pymaid.CatmaidInstance(
        server=OCTO_URL,
        api_token=OCTO_TOKEN,
        http_user="smohinta",
        http_password="headset-recovery-handshake",
        project_id=OCTO_PROJECT # Update to your specific project ID
    )
    
    # Run the sync function
    sync_pk_volumes(TARGET_FOLDER, rm)
