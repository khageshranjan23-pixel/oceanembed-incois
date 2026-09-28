import sys
import os
from huggingface_hub import HfApi

if len(sys.argv) < 3:
    print("Usage: python deploy_hf.py <HF_TOKEN> <SPACE_NAME>")
    sys.exit(1)

token = sys.argv[1]
space_name = sys.argv[2]
api = HfApi(token=token)

try:
    username = api.whoami()["name"]
    repo_id = f"{username}/{space_name}"
    
    print(f"Creating Space {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    
    print("Uploading files...")
    # Upload web/ as static, outputs/public as data, and api.py
    # We will upload the entire oceanembed-kit, but ignore venv and runs/
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["*.venv*", "*runs/*", "*.git*", "*.zip", "cloudflared", "*.log", "data/*"]
    )
    print(f"Successfully deployed to https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Deployment failed: {e}")
