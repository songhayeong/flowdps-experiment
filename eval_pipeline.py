import argparse
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms

import lpips
from pytorch_msssim import ssim


TASK_SPECS = {
    "cs_walshhadamard": 1,
    "cs_blockbased": 4,
    "inpainting": 1,
    "denoising": 1,
    "colorization": 1,
    "sr_avgpool": 12,
    "sr_bicubic": 12,
    "deblur_uni": 9,
    "deblur_gauss": 9,
    "deblur_aniso": 9,
    "deblur_motion": 61,
    "deblur_nonuniform": 33,
    "sr_avgpool_gen": 12,
}

# Some tasks require specific image sizes (e.g., Walsh-Hadamard needs power-of-two).
TASK_IMG_SIZE_OVERRIDES = {
    "cs_walshhadamard": 512,
}


def _list_images(root: Path) -> list[Path]:
    """Return sorted image files (png/jpg/jpeg) from a directory."""
    return sorted(root.glob("*.png")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg")) + sorted(root.glob("*.JPEG"))


def _prepare_fixed_dataset(src_dir: Path, dst_dir: Path, num_images: int) -> list[Path]:
    """Copy the first N images into a fixed dataset directory for reproducible inputs."""
    src_images = _list_images(src_dir)
    if len(src_images) < num_images:
        raise ValueError(f"Need at least {num_images} images in {src_dir}, found {len(src_images)}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = src_images[:num_images]
    copied = []
    for idx, src in enumerate(selected):
        ext = src.suffix.lower()
        dst = dst_dir / f"{idx:04d}{ext}"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _run_sampling(
    task: str,
    deg_scale: int,
    dataset_dir: Path,
    workdir: Path,
    num_images: int,
    meas_seed: int,
    sample_seed: int,
    noise_std: float,
    prompt: str,
    method: str,
    img_size: int,
    imgH: int,
    imgW: int,
    efficient_memory: bool,
):
    """Run the solver subprocess for a task and write outputs under the provided workdir."""
    if task == "sr_avgpool_gen":
        # solve_arbitrary.py may not have meas_seed/sample_seed split; keep old seed for now.
        # If you also modified solve_arbitrary.py similarly, you can mirror the arguments here.
        cmd = [
            "python",
            "solve_arbitrary.py",
            "--seed",
            str(sample_seed),
            "--imgH",
            str(imgH),
            "--imgW",
            str(imgW),
            "--img_path",
            str(dataset_dir),
            "--prompt",
            prompt,
            "--num_samples",
            str(num_images),
            "--task",
            task,
            "--deg_scale",
            str(deg_scale),
            "--method",
            method,
            "--workdir",
            str(workdir),
        ]
    else:
        cmd = [
            "python",
            "solve.py",
            "--meas_seed",
            str(meas_seed),
            "--sample_seed",
            str(sample_seed),
            "--noise_std",
            str(noise_std),
            "--img_size",
            str(img_size),
            "--img_path",
            str(dataset_dir),
            "--prompt",
            prompt,
            "--num_samples",
            str(num_images),
            "--task",
            task,
            "--deg_scale",
            str(deg_scale),
            "--method",
            method,
            "--workdir",
            str(workdir),
        ]

    if efficient_memory:
        cmd.append("--efficient_memory")
    subprocess.run(cmd, check=True)


def _load_residuals(residuals_path: Path) -> dict:
    """Load per-image measurement residuals from a solver run (if present)."""
    if not residuals_path.exists():
        return {}
    with residuals_path.open("r") as f:
        data = json.load(f)
    return {item["image"]: item["residual_l2"] for item in data}


def _compute_metrics(label_dir: Path, recon_dirs: list[Path], residual_maps: list[dict], device: str) -> dict:
    """Compute PSNR/SSIM/LPIPS (and optional residual/diversity) for reconstructions."""
    label_paths = _list_images(label_dir)
    recon_lists = [_list_images(recon_dir) for recon_dir in recon_dirs]
    if any(len(label_paths) != len(recon_paths) for recon_paths in recon_lists):
        raise ValueError("Label/recon count mismatch across samples.")

    # For SSIM: prefer 0..1 inputs.
    to_tensor = transforms.ToTensor()

    # LPIPS expects NCHW in [-1, 1]
    to_lpips = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    per_image = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    residual_values = []
    diversity_values = []

    with torch.no_grad():
        for idx, label_path in enumerate(label_paths):
            label_img = Image.open(label_path).convert("RGB")

            # PSNR on HWC in [0,255]
            label_np = np.asarray(label_img).astype(np.float32)

            # SSIM on 0..1
            label_t = to_tensor(label_img).unsqueeze(0)

            # LPIPS on NCHW in [-1,1]
            label_lp = to_lpips(label_img).unsqueeze(0).to(device)

            sample_metrics = []
            sample_lpips_inputs = []

            for sample_idx, recon_paths in enumerate(recon_lists):
                recon_path = recon_paths[idx]
                recon_img = Image.open(recon_path).convert("RGB")

                recon_np = np.asarray(recon_img).astype(np.float32)
                recon_t = to_tensor(recon_img).unsqueeze(0)

                psnr_value = psnr(label_np, recon_np, data_range=255)
                ssim_value = ssim(label_t, recon_t, data_range=1.0).item()

                recon_lp = to_lpips(recon_img).unsqueeze(0).to(device)
                lpips_value = lpips_fn(label_lp, recon_lp).item()

                residual = residual_maps[sample_idx].get(recon_path.name)
                if residual is not None:
                    residual_values.append(residual)

                sample_metrics.append(
                    {
                        "image": recon_path.name,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "lpips": lpips_value,
                        "residual_l2": residual,
                    }
                )

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                lpips_values.append(lpips_value)
                sample_lpips_inputs.append(recon_lp)

            per_image.append({"image": label_path.name, "samples": sample_metrics})

            # Pairwise LPIPS diversity for the same input image across samples.
            if len(sample_lpips_inputs) > 1:
                pairwise = []
                for i, j in combinations(range(len(sample_lpips_inputs)), 2):
                    pairwise.append(lpips_fn(sample_lpips_inputs[i], sample_lpips_inputs[j]).item())
                diversity_values.append(float(np.mean(pairwise)))

    metrics = {
        "per_image": per_image,
        "mean": {
            "psnr": float(np.mean(psnr_values)),
            "ssim": float(np.mean(ssim_values)),
            "lpips": float(np.mean(lpips_values)),
            "residual_l2": float(np.mean(residual_values)) if residual_values else None,
        },
    }
    if diversity_values:
        metrics["diversity"] = {
            "lpips_pairwise_mean": float(np.mean(diversity_values)),
            "per_image": diversity_values,
        }
    return metrics


def main():
    """Run the full evaluation pipeline across tasks and write per-task results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("samples/imagenet"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_images", type=int, default=50)

    # Base seed used to derive meas_seed/sample_seed
    parser.add_argument("--seed", type=int, default=0)

    # Fix measurement generation across samples; only sample_seed changes.
    parser.add_argument("--meas_seed", type=int, default=None)

    # Measurement noise (keep consistent across tasks if you want pure taxonomy)
    parser.add_argument("--noise_std", type=float, default=0.03)

    parser.add_argument("--prompt", type=str, default="a high quality photo of an object")
    parser.add_argument("--method", type=str, default="flowdps")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--imgH", type=int, default=768)
    parser.add_argument("--imgW", type=int, default=768)

    parser.add_argument("--samples_per_image", type=int, default=1)

    # IMPORTANT: default empty; explicitly pass tasks you want multi-samples for.
    parser.add_argument("--stochastic_tasks", type=str, nargs="*", default=[])

    parser.add_argument("--efficient_memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.meas_seed is None:
        args.meas_seed = args.seed

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = results_dir / f"dataset_{args.num_images}"
    _prepare_fixed_dataset(args.dataset_dir, dataset_dir, args.num_images)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for task, deg_scale in TASK_SPECS.items():

        task_dir = results_dir / task
        images_dir = task_dir / "images"
        workdir = task_dir / "workdir"

        if task == "deblur_nonuniform":
            masks_path = Path("functions/nonuniform/masks/000001.npy")
            kernels_path = Path("functions/nonuniform/kernels/000001.npy")
            if not masks_path.is_file() or not kernels_path.is_file():
                print("Skip deblur_nonuniform: missing nonuniform masks/kernels")
                continue

        if images_dir.exists() and any(images_dir.iterdir()) and not args.overwrite:
            print(f"Skip {task}: {images_dir} already has files. Use --overwrite to re-run.")
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        # Use per-task image size if required.
        task_img_size = TASK_IMG_SIZE_OVERRIDES.get(task, args.img_size)
        task_imgH = args.imgH
        task_imgW = args.imgW
        if task == "sr_avgpool_gen":
            task_imgH = task_img_size
            task_imgW = task_img_size

        # Use multiple samples only for tasks explicitly marked stochastic.
        num_samples = args.samples_per_image if task in args.stochastic_tasks else 1

        recon_dirs = []
        residual_maps = []
        label_dir = None

        for sample_idx in range(num_samples):
            sample_workdir = workdir / f"sample_{sample_idx:02d}"

            # meas_seed is fixed; sample_seed varies across samples.
            meas_seed = args.meas_seed
            sample_seed = args.seed + sample_idx

            _run_sampling(
                task=task,
                deg_scale=deg_scale,
                dataset_dir=dataset_dir,
                workdir=sample_workdir,
                num_images=args.num_images,
                meas_seed=meas_seed,
                sample_seed=sample_seed,
                noise_std=args.noise_std,
                prompt=args.prompt,
                method=args.method,
                img_size=task_img_size,
                imgH=task_imgH,
                imgW=task_imgW,
                efficient_memory=args.efficient_memory,
            )

            recon_dir = sample_workdir / "recon"
            label_dir = sample_workdir / "label"
            if not recon_dir.exists() or not label_dir.exists():
                raise FileNotFoundError(f"Missing recon/label outputs in {sample_workdir}")
            recon_dirs.append(recon_dir)
            residual_maps.append(_load_residuals(sample_workdir / "residuals.json"))

            # Copy recon images to a flat folder for quick browsing.
            for recon_path in _list_images(recon_dir):
                renamed = recon_path.stem + f"_s{sample_idx:02d}" + recon_path.suffix
                shutil.copy2(recon_path, images_dir / renamed)

        metrics = _compute_metrics(label_dir, recon_dirs, residual_maps, device)
        metrics.update(
            {
                "task": task,
                "deg_scale": deg_scale,
                "num_images": args.num_images,
                "samples_per_image": num_samples,
                "seed": args.seed,
                "meas_seed": args.meas_seed,
                "noise_std": args.noise_std,
                "method": args.method,
                "prompt": args.prompt,
                "img_size": task_img_size,
            }
        )

        metrics_path = task_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
