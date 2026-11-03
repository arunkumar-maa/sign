"""
SignBridge AI — Remote Dataset Inspector.
Queries Mendeley Data and Hugging Face to inspect available archives, files, and metadata.
"""

import urllib.request
import re
import json

def inspect_mendeley():
    print("=== INSPECTING MENDELEY (dtg9s785v8/1) ===")
    url = "https://data.mendeley.com/datasets/dtg9s785v8/1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
            title = re.findall(r"<title>(.*?)</title>", html)
            print("Title:", title)
            # Find direct download URLs or zip files
            download_links = set(re.findall(r'href="([^"]+download[^"]*)"', html))
            direct_files = set(re.findall(r'href="([^"]+\.zip[^"]*)"', html))
            print("Download links found:", download_links or direct_files)
            # Look for JSON payload embedded in page
            json_blobs = re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL)
            for blob in json_blobs:
                if "files" in blob and "dtg9s785v8" in blob:
                    print("Found dataset metadata blob!")
                    try:
                        parsed = json.loads(blob)
                        print("Keys in parsed blob:", list(parsed.keys()))
                    except:
                        pass
    except Exception as e:
        print("Mendeley inspect error:", e)

def inspect_hf_fingerspelling():
    print("\n=== INSPECTING HF FINGERSPELLING (kirandevraj/ISL-Fingerspelling) ===")
    api_url = "https://huggingface.co/api/datasets/kirandevraj/ISL-Fingerspelling"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            print("Dataset ID:", info.get("id"))
            print("Author:", info.get("author"))
            print("Description:", (info.get("description") or "")[:200])
            print("Tags:", info.get("tags"))
            siblings = info.get("siblings", [])
            print(f"Total files in repo: {len(siblings)}")
            for s in siblings[:25]:
                print("  File:", s.get("rfilename"))
    except Exception as e:
        print("HF Fingerspelling inspect error:", e)

def inspect_hf_numbers():
    print("\n=== INSPECTING HF NUMBERS (silentone0725/Indian_Sign_Language_Data.gov_Rencoded) ===")
    api_url = "https://huggingface.co/api/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            print("Dataset ID:", info.get("id"))
            siblings = info.get("siblings", [])
            print(f"Total files in repo: {len(siblings)}")
            num_files = [s.get("rfilename") for s in siblings if "number" in s.get("rfilename", "").lower() or any(f"/{i}/" in s.get("rfilename") or f"_{i}." in s.get("rfilename") for i in range(10))]
            print(f"Candidate number files/folders: {len(num_files)}")
            for f in siblings[:30]:
                print("  Sample file:", f.get("rfilename"))
    except Exception as e:
        print("HF Numbers inspect error:", e)

if __name__ == "__main__":
    inspect_mendeley()
    inspect_hf_fingerspelling()
    inspect_hf_numbers()

