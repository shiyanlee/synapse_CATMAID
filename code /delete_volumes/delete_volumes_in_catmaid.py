"""
Code by: Samia Mohinta
Affiliation: University of Cambridge, UK
"""
import pymaid
import requests

SERVER_URL = "https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem/" 
API_TOKEN = "x" #"90f00590555eb256f256d1062e38521cbe180293"
HTTP_USER = "x"
HTTP_PASS = "x"
PROJECT_ID = 32
# Connect to your CATMAID instance
rm = pymaid.CatmaidInstance(
    server=SERVER_URL,
    api_token=API_TOKEN,
    project_id=PROJECT_ID,
    http_user=HTTP_USER,
    http_password=HTTP_PASS
)

# Fetch all volumes
volumes = pymaid.get_volume(remote_instance=rm)
print(volumes.columns.tolist())  # inspect available columns

# Filter for PK_* volumes by shiyan
to_delete = volumes[
    volumes["name"].str.startswith("PK_") &
    (volumes["user_id"] == 38)  # column name may be 'user' or 'user_id'
    ]

print(f"Found {len(to_delete)} volumes to delete:")
print(to_delete[["id", "name", "user_id"]])

# verify prior to deleting. Wont work if the volume is not created by you. Use sudo to delete volumes uploaded by other people.
# for _, row in to_delete.iterrows():
#     pymaid.delete_volume(row["id"], no_prompt=True, remote_instance=rm)
#     print(f"✓ Deleted: {row['name']} (ID: {row['id']})")

# Use a requests session with Basic auth to delete
session = requests.Session()
session.auth = (HTTP_USER, HTTP_PASS)

# Hit the server once to get session cookie + CSRF token
session.get(SERVER_URL)
csrf_token = next((v for k, v in session.cookies.items() if k.startswith("csrftoken")), None)
print(f"CSRF token: {csrf_token}")

for _, row in to_delete.iterrows():
    vol_id = int(row["id"])
    vol_name = row["name"]

    resp = session.delete(
        f"{SERVER_URL}{PROJECT_ID}/volumes/{vol_id}/",
        headers={
            "X-Authorization": f"Token {API_TOKEN}",
            "X-CSRFToken": csrf_token,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{SERVER_URL}{PROJECT_ID}/volumes/{vol_id}/",
        }
    )

    if resp.status_code == 200:
        print(f"✓ Deleted: {vol_name} (ID: {vol_id})")
    else:
        print(f"✗ Failed: {vol_name} (ID: {vol_id}) — {resp.status_code}: {resp.text[:300]}")
