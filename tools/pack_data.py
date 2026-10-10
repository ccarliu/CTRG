"""
Build the released data bundle ($CTRG_DATA_ROOT, default <repo>/data) from the
original locations, and optionally pack it into a single archive for upload.

Layout produced (must match paths_config.py):

    data/
      ctrate/
        reports/  labels/  region_report/  metadata/
        report_hierarchy_tree/{train,valid}/
        mask_fast/{train,valid}/
      text_embedding/ipmiversion/

Usage:
    # 1) assemble data/ with symlinks (no extra disk; code runs immediately)
    python tools/pack_data.py
    # 2) verify every bundled path exists and count files
    python tools/pack_data.py --check
    # 3) produce an uploadable archive (follows symlinks)
    python tools/pack_data.py --tar ctrg_data.tar.gz
    # or copy real files instead of symlinking
    python tools/pack_data.py --mode copy --dest /path/to/release/data
"""
import argparse
import os
import shutil
import sys
import tarfile

# Source roots are passed with --ctrate-src (dir holding the CT-RATE CSVs,
# report_hierarchy_tree/ and data_volumes/mask_processed/) and --m3ae-src (dir
# holding text_embedding/ipmiversion/).

# (destination relative to data root, source root key, source relative path)
MANIFEST = [
    ("ctrate/reports/train_reports.csv", "ctrate", "train_reports.csv"),
    ("ctrate/reports/dataset_radiology_text_reports_validation_reports.csv", "ctrate",
     "dataset_radiology_text_reports_validation_reports.csv"),
    ("ctrate/labels/dataset_multi_abnormality_labels_train_predicted_labels.csv", "ctrate",
     "dataset_multi_abnormality_labels_train_predicted_labels.csv"),
    ("ctrate/labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv", "ctrate",
     "dataset_multi_abnormality_labels_valid_predicted_labels.csv"),
    ("ctrate/region_report/train_region_report_abnormality_exacted_shorted.csv", "ctrate",
     "train_region_report_abnormality_exacted_shorted.csv"),
    ("ctrate/region_report/dataset_radgenome_files_validation_region_report_abnormalities_extracted_shorted0_60000.csv",
     "ctrate", "dataset_radgenome_files_validation_region_report_abnormalities_extracted_shorted0_60000.csv"),
    ("ctrate/metadata/train_metadata.csv", "ctrate", "train_metadata.csv"),
    ("ctrate/metadata/dataset_metadata_validation_metadata.csv", "ctrate",
     "dataset_metadata_validation_metadata.csv"),
    ("ctrate/report_hierarchy_tree/train", "ctrate", "report_hierarchy_tree/train"),
    ("ctrate/report_hierarchy_tree/valid", "ctrate", "report_hierarchy_tree/valid"),
    ("ctrate/mask_fast/train", "ctrate", "data_volumes/mask_processed/train_fast"),
    ("ctrate/mask_fast/valid", "ctrate", "data_volumes/mask_processed/valid_fast"),
    ("text_embedding/ipmiversion", "m3ae", "text_embedding/ipmiversion"),
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_sources(args):
    roots = {"ctrate": args.ctrate_src, "m3ae": args.m3ae_src}
    return [(dst, os.path.join(roots[key], rel)) for dst, key, rel in MANIFEST]


def _count(path):
    if os.path.isdir(path):
        return sum(len(files) for _, _, files in os.walk(path, followlinks=True))
    return 1 if os.path.exists(path) else 0


def assemble(entries, dest, mode, force):
    missing = [src for _, src in entries if not os.path.exists(src)]
    if missing:
        print("[ERROR] missing sources:\n  " + "\n  ".join(missing))
        sys.exit(1)
    for rel, src in entries:
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.lexists(out):
            if not force:
                print(f"[skip] exists: {out}")
                continue
            if os.path.isdir(out) and not os.path.islink(out):
                shutil.rmtree(out)
            else:
                os.remove(out)
        if mode == "symlink":
            os.symlink(src, out)
        elif os.path.isdir(src):
            shutil.copytree(src, out)
        else:
            shutil.copy2(src, out)
        print(f"[{mode}] {rel}")


def check(entries, dest):
    ok = True
    for rel, _ in entries:
        out = os.path.join(dest, rel)
        n = _count(out)
        status = "OK " if n else "MISSING"
        ok &= bool(n)
        print(f"[{status}] {rel}  ({n} file{'s' if n != 1 else ''})")
    return ok


def make_tar(dest, tar_path):
    arcroot = os.path.basename(os.path.normpath(dest))
    print(f"[tar] {dest} -> {tar_path} (following symlinks, this can take a while)")
    with tarfile.open(tar_path, "w:gz", dereference=True) as tf:
        tf.add(dest, arcname=arcroot)
    print("[tar] done")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=os.environ.get("CTRG_DATA_ROOT", os.path.join(_REPO_ROOT, "data")),
                   help="data root to build (default: $CTRG_DATA_ROOT or <repo>/data)")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--ctrate-src", default=None, help="CT-RATE source root (required unless --check)")
    p.add_argument("--m3ae-src", default=None, help="root containing text_embedding/ipmiversion (required unless --check)")
    p.add_argument("--force", action="store_true", help="overwrite existing entries in --dest")
    p.add_argument("--check", action="store_true", help="only verify the bundle under --dest")
    p.add_argument("--tar", default=None, help="write a .tar.gz of --dest (after assembling)")
    args = p.parse_args()

    if args.check:
        entries = [(dst, None) for dst, _, _ in MANIFEST]
        sys.exit(0 if check(entries, args.dest) else 1)
    if not (args.ctrate_src and args.m3ae_src):
        p.error("--ctrate-src and --m3ae-src are required to assemble the bundle")
    entries = _resolve_sources(args)
    assemble(entries, args.dest, args.mode, args.force)
    if not check(entries, args.dest):
        sys.exit(1)
    if args.tar:
        make_tar(args.dest, args.tar)


if __name__ == "__main__":
    main()
