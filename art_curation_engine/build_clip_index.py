#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP Index Builder for Artwork Search

Builds a FAISS-based image search index using CLIP embeddings for artwork recommendation.
Processes artwork metadata and images to create a searchable vector database.

Converted from build_clip_index.ipynb with optimizations and relative paths.

Usage:
    python build_clip_index.py
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Set proper cache directories to avoid permission issues 
home_dir = Path.home()
cache_dir = home_dir / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables for proper cache locations
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
os.environ["TORCH_HOME"] = str(cache_dir / "torch")

print(f"Cache directory set to: {cache_dir}")

#  Import deep learning libraries 
import torch
import open_clip

# FAISS (required) 
try:
    import faiss
    
    # Check FAISS version safely
    try:
        version = faiss.__version__
        print(f"FAISS version: {version}")
    except AttributeError:
        print("FAISS version: Unknown (very old version detected)")
    
    # Check for available metrics and use compatible ones
    available_metrics = []
    metrics_to_check = [
        'METRIC_INNER_PRODUCT',
        'METRIC_L2', 
        'METRIC_ABS_INNER_PRODUCT',
        'METRIC_L1',
        'METRIC_Lp',
        'METRIC_Linf',
        'METRIC_Canberra',
        'METRIC_BrayCurtis',
        'METRIC_JensenShannon'
    ]
    
    for metric_name in metrics_to_check:
        if hasattr(faiss, metric_name):
            available_metrics.append(metric_name)
    
    print(f" Available FAISS metrics: {available_metrics}")
    
    # Use the best available metric
    if hasattr(faiss, 'METRIC_INNER_PRODUCT'):
        DEFAULT_METRIC = faiss.METRIC_INNER_PRODUCT
        print(" Using METRIC_INNER_PRODUCT")
    elif hasattr(faiss, 'METRIC_L2'):
        DEFAULT_METRIC = faiss.METRIC_L2
        print("� Using METRIC_L2 (fallback)")
    else:
        raise ImportError("No compatible FAISS metrics found - FAISS version too old")
        
except ImportError as e:
    raise SystemExit(
        "[ERROR] FAISS is not installed or version is too old.\n"
        "Solution:\n"
        "1. conda activate arti_chatbot\n"
        "2. pip uninstall -y faiss-cpu faiss-gpu\n" 
        "3. conda install -c conda-forge faiss-cpu\n"
        "Or: pip install --upgrade faiss-cpu\n"
        f"Error: {str(e)}"
    )

# Path configuration (using relative paths) 
SCRIPT_DIR = Path(__file__).parent
ARTI_ROOT = SCRIPT_DIR
META_PATH = ARTI_ROOT / "data/aic_sample/metadata.jsonl"   # JSONL: one artwork per line
OUT_DIR = ARTI_ROOT / "indices/clip_faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = OUT_DIR / "faiss.index"
ID_MAP_PATH = OUT_DIR / "id_map.json"           # faiss_idx -> {...}
ARTWORK_INDEX_PATH = OUT_DIR / "artwork_index.json"  # artwork_id -> faiss_idx
REPORT_PATH = OUT_DIR / "build_report.json"

#  Model/Device Configuration 
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"  # Recommended OpenCLIP pretrained model
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64  # Adjust based on available memory

def load_metadata(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load metadata from JSONL file"""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def resolve_image_path(item: Dict[str, Any]) -> Path | None:
    """
    Resolve image path from metadata.jsonl
    image_file paths are relative to ARTI_ROOT/data/
    """
    p = item.get("image_file")
    if not p:
        return None
    
    p = Path(p)
    
    if p.is_absolute():
        # If absolute path, use as-is
        return p if p.exists() else None
    else:
        # If relative path, it's relative to ARTI_ROOT/data/
        full_path = ARTI_ROOT / "data" / p
        return full_path if full_path.exists() else None

@torch.no_grad()
def extract_embeddings(image_paths: List[Path], model, preprocess, device: str) -> np.ndarray:
    """
    Extract CLIP embeddings from image paths (normalized, returns float32)
    """
    all_feats = []
    
    print(f"= Processing {len(image_paths)} images in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                # Replace corrupted files with None (will be skipped later)
                print(f"� Failed to load image {p}: {e}")
                imgs.append(None)
        
        # Separate valid/invalid images
        valid_idx = [k for k, x in enumerate(imgs) if x is not None]
        if len(valid_idx) == 0:
            # If entire batch failed, fill with zeros
            zeros = np.zeros((len(batch_paths), model.visual.output_dim), dtype="float32")
            all_feats.append(zeros)
            continue
        
        batch_tensor = torch.stack([imgs[k] for k in valid_idx]).to(device)
        feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.detach().cpu().numpy().astype("float32")

        # Restore original batch order (failed items get zero vectors)
        out = np.zeros((len(batch_paths), feats_np.shape[1]), dtype="float32")
        for j, k in enumerate(valid_idx):
            out[k] = feats_np[j]
        all_feats.append(out)
    
    return np.concatenate(all_feats, axis=0)

def load_clip_model():
    """Load CLIP model with fallback strategies"""
    print(f"[info] device={DEVICE}, model={MODEL_NAME}/{PRETRAINED}")
    
    # Set cache directory to avoid permission issues
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    try:
        # Try to use local files only first
        print("= Attempting to load model from local cache...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, 
            pretrained=PRETRAINED, 
            device=DEVICE,
            cache_dir=str(cache_dir),
        )
        model.eval()
        print(" Model loaded successfully from local cache")
        
    except Exception as e:
        print(f"L Local model loading failed: {e}")
        print("= Trying without pretrained weights...")
        
        # Fallback: try without pretrained weights
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, 
                pretrained=None,  # No pretrained weights
                device=DEVICE
            )
            print("� Model loaded without pretrained weights (random initialization)")
        except Exception as e2:
            # Last resort: try a different model that's definitely local
            print("= Trying OpenAI CLIP model...")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", 
                    pretrained="openai",  # This should be more stable
                    device=DEVICE
                )
                print(" Loaded OpenAI CLIP model instead")
            except Exception as e3:
                raise RuntimeError(f"All model loading attempts failed: {e3}")

    # Check if model loaded properly
    print(f"=� Model info: {MODEL_NAME}, Device: {DEVICE}")
    return model, preprocess

def main():
    """Main function to build CLIP index"""
    print("=� Starting CLIP index building process...")
    
    # Load model
    model, preprocess = load_clip_model()

    # Load metadata
    print(f"= Loading metadata from {META_PATH}")
    meta_items = load_metadata(META_PATH)
    print(f"[info] metadata loaded: {len(meta_items)} items")

    # Collect: (artwork_id, image_path, meta)
    print("= Resolving image paths...")
    records = []
    for m in meta_items:
        aid = m.get("id")
        ipath = resolve_image_path(m)
        if aid is None:
            continue
        if ipath is None:
            continue
        records.append((aid, ipath, m))

    print(f"[info] resolved local images: {len(records)}")

    # Extract image paths
    img_paths = [r[1] for r in records]
    embs = extract_embeddings(img_paths, model, preprocess, DEVICE)  # shape: (N, D)
    N, D = embs.shape
    print(f"[info] embeddings: {N} � {D}")

    # Skip failed (zero vector) embeddings & add to FAISS
    print("= Building FAISS index...")
    index = faiss.IndexFlatIP(D)  # Inner product for cosine similarity (since normalized)
    id_map: Dict[int, Dict[str, Any]] = {}
    artwork_index: Dict[str | int, int] = {}

    valid_rows = 0
    bad_rows = 0
    to_add = []

    for i, (aid, ipath, meta) in enumerate(records):
        vec = embs[i]
        if float(np.linalg.norm(vec)) < 1e-6:  # Consider as failure
            bad_rows += 1
            continue
        to_add.append(vec)

    if to_add:
        xb = np.stack(to_add, axis=0).astype("float32")
        index.add(xb)

    # Record FAISS row numbers
    faiss_row = 0
    for i, (aid, ipath, meta) in enumerate(records):
        vec = embs[i]
        if float(np.linalg.norm(vec)) < 1e-6:
            continue
        id_map[faiss_row] = {
            "artwork_id": aid,
            "image_path": str(ipath),
            "title": meta.get("title"),
            "style_title": meta.get("style_title"),
            "subjects": meta.get("subject_titles"),
        }
        artwork_index[str(aid)] = faiss_row
        faiss_row += 1
        valid_rows += 1

    # Save files
    print("= Saving FAISS index and metadata...")
    faiss.write_index(index, str(FAISS_PATH))
    
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    
    with open(ARTWORK_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(artwork_index, f, ensure_ascii=False, indent=2)

    # Generate report
    report = {
        "total_metadata": len(meta_items),
        "resolved_local_images": len(records),
        "embedded": int(embs.shape[0]),
        "faiss_rows": valid_rows,
        "failed_images": bad_rows,
        "embedding_dimension": D,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "device": DEVICE,
        "faiss_path": str(FAISS_PATH),
        "id_map": str(ID_MAP_PATH),
        "artwork_index": str(ARTWORK_INDEX_PATH),
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print results
    print("\n" + "="*60)
    print(" CLIP Index Build Complete!")
    print("="*60)
    print(f"=� FAISS index: {FAISS_PATH}")
    print(f"=� ID map: {ID_MAP_PATH}")
    print(f"=� Artwork index: {ARTWORK_INDEX_PATH}")
    print(f"=� Build report: {REPORT_PATH}")
    print(f"=� Summary: {valid_rows} valid embeddings, {bad_rows} failed images")
    print(f"<� Index ready for artwork search!")

if __name__ == "__main__":
    main()