import os, glob, numpy as np

DATA_DIR = "./data/fullres/BraTS2020_NPZ"  # 改成你的目录（BRaTS同理）
npz_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

bad = []
axis_swap = []
ok = []
for npz in npz_list:
    base = npz[:-4]
    img_p = base + ".npy"
    seg_p = base + "_seg.npy"
    if not (os.path.exists(img_p) and os.path.exists(seg_p)):
        print("[MISS]", os.path.basename(npz))
        continue

    img = np.load(img_p, "r+")
    seg = np.load(seg_p, "r+")
    # 期望 image 是 4 通道
    if img.ndim != 4:
        print("[IMG_NDIM != 4]", os.path.basename(npz), img.shape)
        continue

    # 统一把图像转成 (4, D, H, W) 来拿出 DHW
    if img.shape[0] == 4:
        # (4, D/H, H/W, W/D) -> 如果最后一维像 155 就是 (4,H,W,D)
        if img.shape[-1] < 200 and img.shape[-1] >= 50:
            D, H, W = img.shape[-1], img.shape[1], img.shape[2]
        else:
            # 默认当作 (4,D,H,W)
            D, H, W = img.shape[1], img.shape[2], img.shape[3]
    elif img.shape[-1] == 4:
        # (D,H,W,4)
        D, H, W = img.shape[0], img.shape[1], img.shape[2]
    else:
        print("[IMG_CH != 4]", os.path.basename(npz), img.shape)
        continue

    # 检查 seg
    if seg.ndim == 3:
        if seg.shape == (D, H, W):
            ok.append(npz)
        elif seg.shape == (H, W, D):
            axis_swap.append(npz)
        elif seg.shape[::-1] == (D, H, W):
            axis_swap.append(npz)
        else:
            print("[SEG_3D_MISMATCH]", os.path.basename(npz), "seg:", seg.shape, "expect:", (D,H,W))
            bad.append(npz)
    elif seg.ndim == 2:
        print("[SEG_2D]", os.path.basename(npz), seg.shape, "should be 3D (D,H,W) =", (D,H,W))
        bad.append(npz)
    elif seg.ndim == 4 and seg.shape[0] == 1:
        # (1, H, W, D) 或 (1, D, H, W) 这种也记下来
        print("[SEG_4D_WITH_1]", os.path.basename(npz), seg.shape)
        bad.append(npz)
    else:
        print("[SEG_UNSUPPORTED]", os.path.basename(npz), seg.shape)
        bad.append(npz)

print("\n=== SUMMARY ===")
print("OK (3D aligned):", len(ok))
print("Need axis swap (HWD->DHW):", len(axis_swap))
print("Bad (2D or mismatch):", len(bad))
if axis_swap[:5]:
    print("  e.g. axis swap:", [os.path.basename(x) for x in axis_swap[:5]])
if bad[:5]:
    print("  e.g. bad:", [os.path.basename(x) for x in bad[:5]])