import os
import glob
import argparse
from datetime import datetime

import numpy as np
import h5py


BUNDLE_TYPE = "noise_models_bundle_v1"
DEFAULT_NOISE_NN_H5 = r"D:\4.DATASETS\NOISE_MODELS_CH3OCHO_NN_GUAPOS_v10plotstyle_roi"
DEFAULT_OUTPUT_H5 = r"D:\4.DATASETS\NOISE_MODELS_CH3OCHO_NN_GUAPOS_v10plotstyle_roi\NOISE_MODELS_CH3OCHO_NN_GUAPOS_v9plotstyle_roi_bundle.h5"


def _default_output_h5(input_dir: str) -> str:
    src_dir = os.path.abspath(str(input_dir))
    base_name = os.path.basename(src_dir.rstrip("\\/")) or "noise_models"
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(src_dir, f"{base_name}_bundle_{stamp}.h5")


def _read_cfg_json_text(noise_model_h5: str) -> str:
    with h5py.File(noise_model_h5, "r") as hf:
        cfg_raw = hf["config_json"][()]
    if isinstance(cfg_raw, (bytes, bytearray, np.bytes_)):
        return cfg_raw.decode("utf-8")
    return str(cfg_raw)


def build_noise_bundle(input_dir: str, output_h5: str, compression: str = "gzip", compression_level: int = 6):
    src_dir = os.path.abspath(str(input_dir))
    out_file = os.path.abspath(str(output_h5))

    noise_files = sorted(glob.glob(os.path.join(src_dir, "**", "final_noise_model.h5"), recursive=True))
    noise_files = [p for p in noise_files if os.path.isfile(p)]
    if not noise_files:
        raise FileNotFoundError(f"No final_noise_model.h5 files found under: {src_dir}")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    total_in_bytes = sum(os.path.getsize(p) for p in noise_files)

    if os.path.isfile(out_file):
        os.remove(out_file)

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_file, "w") as hf:
        hf.attrs["bundle_type"] = BUNDLE_TYPE
        hf.attrs["created_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        hf.attrs["source_dir"] = src_dir
        hf.attrs["n_models"] = int(len(noise_files))

        grp_models = hf.create_group("noise_models")
        grp_cfg = hf.create_group("noise_cfg_json")

        for i, path in enumerate(noise_files, start=1):
            name = f"model_{i:05d}"
            rel_path = os.path.relpath(path, src_dir).replace("\\", "/")

            with open(path, "rb") as f:
                blob = f.read()
            arr = np.frombuffer(blob, dtype=np.uint8)

            ds = grp_models.create_dataset(
                name,
                data=arr,
                compression=compression,
                compression_opts=int(compression_level),
                shuffle=True,
            )
            ds.attrs["rel_path"] = rel_path

            cfg_text = _read_cfg_json_text(path)
            grp_cfg.create_dataset(name, data=np.array(cfg_text, dtype=str_dt))

            print(f"[{i}/{len(noise_files)}] packed: {rel_path}")

    out_bytes = os.path.getsize(out_file)
    ratio = (float(out_bytes) / float(total_in_bytes)) if total_in_bytes > 0 else 1.0

    print("\nDone")
    print(f"Input directory : {src_dir}")
    print(f"Output bundle   : {out_file}")
    print(f"Models packed   : {len(noise_files)}")
    print(f"Input size      : {total_in_bytes / (1024**3):.3f} GB")
    print(f"Output size     : {out_bytes / (1024**3):.3f} GB")
    print(f"Compression     : {ratio:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Pack NOISE_MODELS folder into a single .h5 bundle")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_NOISE_NN_H5,
        help=f"Root folder containing final_noise_model.h5 files (default: {DEFAULT_NOISE_NN_H5})",
    )
    parser.add_argument(
        "output_h5",
        nargs="?",
        default=DEFAULT_OUTPUT_H5,
        help=f"Output .h5 bundle file path (default: {DEFAULT_OUTPUT_H5})",
    )
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf"], help="HDF5 compression")
    parser.add_argument("--level", type=int, default=6, help="Compression level (used by gzip)")
    args = parser.parse_args()

    build_noise_bundle(
        input_dir=args.input_dir,
        output_h5=args.output_h5,
        compression=args.compression,
        compression_level=args.level,
    )


if __name__ == "__main__":
    main()
