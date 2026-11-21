"""Upload the trained model folder to Hugging Face.

Log in once with YOUR OWN token (never put the token in the code or in git):
    hf auth login
Then:
    python src/push_to_hub.py --repo <your-hf-username>/xlmr-kazakh-profanity
"""
import argparse

from huggingface_hub import HfApi

from predict import DEFAULT_MODEL

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
args = parser.parse_args()

api = HfApi()
api.create_repo(args.repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=DEFAULT_MODEL, repo_id=args.repo, repo_type="model")
print(f"uploaded -> https://huggingface.co/{args.repo}")
