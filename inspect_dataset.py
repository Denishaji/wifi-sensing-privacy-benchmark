import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/6367411"
OUTPUT = Path(__file__).resolve().parent / "dataset_manifest.json"


def main():
    request = urllib.request.Request(
        RECORD_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "wifi-privacy-reproduction/0.1",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            record = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"Dataset server returned HTTP {error.code}.")
        if error.code == 429:
            print("The server is rate-limiting requests. Try again later.")
        return 1
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"Could not retrieve the dataset listing: {error}")
        return 1

    files = record.get("files", [])
    if not files:
        print("No files returned. No recordings downloaded.")
        return 1

    manifest = {
        "source": RECORD_URL,
        "title": record.get("metadata", {}).get("title"),
        "license": record.get("metadata", {}).get("license"),
        "files": [
            {
                "name": item["key"],
                "bytes": item["size"],
                "checksum": item.get("checksum"),
                "links": item.get("links", {}),
            }
            for item in files
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(manifest["title"])
    for item in manifest["files"]:
        size_mb = item["bytes"] / 1_000_000
        print(f"{size_mb:10.1f} MB  {item['name']}")

    print(f"\nMetadata saved to {OUTPUT}")
    print("No recording archives were downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())