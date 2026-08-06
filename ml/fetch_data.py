"""Download the official dair-ai/emotion split (16k / 2k / 2k) as parquet.

The notebook used `nlp.load_dataset('emotion')`, which pinned the same split;
the `nlp` package has since been renamed and the loader script removed, so we
pull the files straight from the Hugging Face CDN instead. No account needed.
"""

import os
import urllib.request

BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "validation", "test"):
        dst = os.path.join(OUT, f"{split}.parquet")
        if os.path.exists(dst):
            print(f"{split}: cached")
            continue
        url = f"{BASE}/{split}-00000-of-00001.parquet"
        print(f"{split}: {url}")
        urllib.request.urlretrieve(url, dst)
    print("done ->", OUT)
