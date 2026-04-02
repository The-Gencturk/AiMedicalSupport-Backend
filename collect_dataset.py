# collect_dataset.py - tamamen değiştir
import requests
import os
from pathlib import Path
import time
import random

DATASET_DIR = "AiModels/Classifier/dataset"

searches = {
    "brain":   ["brain+CT+scan+axial", "head+CT+scan", "cranial+tomography", "beyin+tomografi"],
    "lung":    ["chest+CT+scan", "lung+tomography", "pulmonary+CT", "akciger+tomografi"],
    "abdomen": ["abdominal+CT+scan", "stomach+CT", "liver+CT+scan", "karin+tomografi"],
    "bone":    ["bone+CT+scan", "spine+CT", "skeletal+tomography", "kemik+tomografi"],
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def download_from_unsplash(keyword, save_dir, count=50):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    downloaded = 0
    page = 1
    while downloaded < count:
        url = f"https://api.unsplash.com/search/photos?query={keyword}&per_page=30&page={page}&client_id=YOUR_KEY"
        # Unsplash key lazım, bu çalışmaz
        page += 1
        if page > 10:
            break
    return downloaded

def download_images_bing(keyword, save_dir, count=100):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    existing = len(list(Path(save_dir).glob("*.jpg")))
    
    offset = 0
    downloaded = 0
    while downloaded < count:
        url = f"https://www.bing.com/images/async?q={keyword}&first={offset}&count=35&mmasync=1"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            import re
            urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', r.text)
            if not urls:
                break
            for img_url in urls:
                if downloaded >= count:
                    break
                try:
                    img_r = requests.get(img_url, timeout=5, headers=headers)
                    if img_r.status_code == 200 and 'image' in img_r.headers.get('content-type', ''):
                        fname = f"{save_dir}/{existing + downloaded + 1:04d}.jpg"
                        with open(fname, 'wb') as f:
                            f.write(img_r.content)
                        downloaded += 1
                except:
                    pass
            offset += 35
            time.sleep(random.uniform(0.5, 1.5))
        except:
            break
    return downloaded

for organ, keywords in searches.items():
    print(f"\n[{organ}] indiriliyor...")
    save_dir = f"{DATASET_DIR}/{organ}"
    total = 0
    for kw in keywords:
        n = download_images_bing(kw, save_dir, count=80)
        total += n
        print(f"  '{kw}': {n} görüntü")
        time.sleep(2)
    print(f"[{organ}] toplam yeni: {total}")

print("\nBitti!")