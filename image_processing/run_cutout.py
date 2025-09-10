#!/usr/bin/env python3
"""
BiRefNet GPU Foreground Cutout Tool
Clips figures from images using BiRefNet with GPU acceleration (CUDA/ROCm).
Output files have '_fg' suffix and transparent backgrounds.
"""

import os
import sys
import argparse
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# === 기본 설정 ===
DEFAULT_MODEL = "./models/BiRefNet-general-epoch_244.onnx"
DEFAULT_FG_DIR = "./out/clipped_images_fg"
DEFAULT_BG_DIR = "./out/clipped_images_bg"
DEFAULT_IN_SIZE = 1024

def make_session(model_path: str) -> ort.InferenceSession:
    """Create ONNX Runtime session with GPU acceleration (CUDA/ROCm)"""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1  # Fix threading issues
    sess_opts.inter_op_num_threads = 1
    
    available_providers = ort.get_available_providers()
    print("[Info] Available providers:", available_providers)
    
    # GPU provider options for optimization
    gpu_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'do_copy_in_default_stream': True,
    }
    
    # Add CUDA-specific options if CUDA is available
    if "CUDAExecutionProvider" in available_providers:
        gpu_provider_options['cudnn_conv_algo_search'] = 'HEURISTIC'
    
    providers = []
    provider_options = []
    
    # Priority order: CUDA -> ROCm -> CPU
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
        provider_options.append(gpu_provider_options)
        print("[Info] CUDA provider will be used")
    elif "ROCMExecutionProvider" in available_providers:
        providers.append("ROCMExecutionProvider")
        provider_options.append(gpu_provider_options)
        print("[Info] ROCm provider will be used")
    
    # Always add CPU as fallback
    providers.append("CPUExecutionProvider")
    provider_options.append({})
    
    sess = ort.InferenceSession(model_path, sess_opts, providers=providers, provider_options=provider_options)
    print("[Info] Using providers:", sess.get_providers())
    return sess

def predict_mask(sess: ort.InferenceSession, img_np_float01: np.ndarray, in_size: int) -> np.ndarray:
    """
    img_np_float01: (H, W, 3), float32, [0,1]
    return: mask float32 (H, W) in [0,1]
    """
    h, w = img_np_float01.shape[:2]
    resized = cv2.resize(img_np_float01, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
    x = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1,3,H,W)

    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x})[0]
    # 출력 형태가 (1,1,H,W) 또는 (1,H,W)일 수 있음
    if out.ndim == 4:
        mask_small = out[0, 0]
    else:
        mask_small = out[0]
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask

def create_fg_bg_images(img_np_uint8: np.ndarray, mask01: np.ndarray, keep_largest=True, feather=5) -> tuple[Image.Image, Image.Image]:
    """
    Create both foreground and background images from the same mask
    img_np_uint8:(H,W,3) uint8, mask01:(H,W) float32 in [0,1]
    Returns: (foreground_rgba, background_rgba)
    """
    h, w = mask01.shape
    alpha = (mask01 * 255.0).astype(np.uint8)

    if keep_largest:
        # 가장 큰 연결 성분만 유지 (작품 배경 잔여 제거에 효과)
        _, binm = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        num, labels = cv2.connectedComponents(binm)
        if num > 2:
            areas = [(labels == i).sum() for i in range(1, num)]
            keep_id = 1 + int(np.argmax(areas))
            keep = (labels == keep_id).astype(np.uint8) * 255
            alpha = cv2.bitwise_and(alpha, keep)

    if feather and feather > 0 and feather % 2 == 1:
        alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)

    # Create foreground (figure) - img * M
    fg_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fg_rgba[:, :, :3] = img_np_uint8
    fg_rgba[:, :, 3] = alpha
    
    # Create background (everything except figure) - img * (1 - M)
    bg_alpha = 255 - alpha  # Invert mask
    bg_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    bg_rgba[:, :, :3] = img_np_uint8
    bg_rgba[:, :, 3] = bg_alpha
    
    return Image.fromarray(fg_rgba), Image.fromarray(bg_rgba)

def cutout_one(sess: ort.InferenceSession, img_path: str, fg_dir: str, bg_dir: str, in_size: int):
    """Process single image and save both foreground and background"""
    im = Image.open(img_path).convert("RGB")
    arr_f = np.asarray(im).astype(np.float32) / 255.0
    mask = predict_mask(sess, arr_f, in_size)
    fg_rgba, bg_rgba = create_fg_bg_images((arr_f * 255).astype(np.uint8), mask, keep_largest=True, feather=5)
    
    # Generate output paths with _fg and _bg suffixes
    base = os.path.splitext(os.path.basename(img_path))[0]
    fg_path = os.path.join(fg_dir, f"{base}_fg.png")
    bg_path = os.path.join(bg_dir, f"{base}_bg.png")
    
    # Create output directories
    os.makedirs(fg_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    
    # Save both images
    fg_rgba.save(fg_path)
    bg_rgba.save(bg_path)
    
    print(f"[OK] Saved FG: {fg_path}")
    print(f"[OK] Saved BG: {bg_path}")
    return fg_path, bg_path

def cutout_batch(sess: ort.InferenceSession, in_dir: str, fg_dir: str, bg_dir: str, in_size: int):
    """Process all images in a directory"""
    os.makedirs(fg_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    names = [n for n in os.listdir(in_dir) if n.lower().endswith(exts)]
    print(f"[Info] Found {len(names)} images in {in_dir}")
    
    success_count = 0
    for n in names:
        ip = os.path.join(in_dir, n)
        try:
            cutout_one(sess, ip, fg_dir, bg_dir, in_size)
            success_count += 1
        except Exception as e:
            print(f"[Warn] Failed: {n} -> {e}")
    
    print(f"[Info] Successfully processed {success_count}/{len(names)} images")
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="BiRefNet GPU Foreground/Background Cutout Tool (CUDA/ROCm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image (saves foo_fg.png and foo_bg.png)
  python run_cutout.py -i foo.jpg
  
  # Batch processing
  python run_cutout.py -b ./input_folder/
  
  # Custom output directories
  python run_cutout.py -i image.jpg --fg-dir ./fg_output/ --bg-dir ./bg_output/
  
  # Custom model
  python run_cutout.py -i image.jpg -m ./custom_model.onnx
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--image', help='Single image file path')
    input_group.add_argument('-b', '--batch', help='Input directory for batch processing')
    
    # Output directories
    parser.add_argument('--fg-dir', default=DEFAULT_FG_DIR,
                       help=f'Foreground output directory (default: {DEFAULT_FG_DIR})')
    parser.add_argument('--bg-dir', default=DEFAULT_BG_DIR,
                       help=f'Background output directory (default: {DEFAULT_BG_DIR})')
    
    # Optional settings
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL,
                       help=f'ONNX model path (default: {DEFAULT_MODEL})')
    parser.add_argument('-s', '--size', type=int, default=DEFAULT_IN_SIZE,
                       help=f'Model input size (default: {DEFAULT_IN_SIZE})')
    
    args = parser.parse_args()
    
    # Validate model exists
    if not os.path.exists(args.model):
        print(f"[Error] Model not found: {args.model}")
        sys.exit(1)
    
    # Create session
    sess = make_session(args.model)
    
    # Process input
    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"[Error] Image not found: {args.image}")
            sys.exit(2)
        cutout_one(sess, args.image, args.fg_dir, args.bg_dir, args.size)
    
    elif args.batch:
        # Batch mode
        if not os.path.isdir(args.batch):
            print(f"[Error] Input directory not found: {args.batch}")
            sys.exit(3)
        cutout_batch(sess, args.batch, args.fg_dir, args.bg_dir, args.size)

if __name__ == "__main__":
    main()