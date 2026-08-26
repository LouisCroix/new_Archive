#!/usr/bin/env bash
# Download ILSVRC2012 from Kaggle and prepare the numeric ImageFolder layout
# expected by imagenet_data.py:
#
#   DATA_ROOT/train/0/*.JPEG ... DATA_ROOT/train/999/*.JPEG
#   DATA_ROOT/val/0/*.JPEG   ... DATA_ROOT/val/999/*.JPEG
#
# Before running this script, sign in to Kaggle, accept the rules at
# https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data,
# and configure Kaggle API credentials.

set -Eeuo pipefail

usage() {
    echo "Usage: $0 /absolute/path/to/imagenet" >&2
    echo "Example: $0 /cis/home/cyang140/datasets/imagenet" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

for command_name in unzip python3 realpath df awk; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
done

TARGET="$(realpath -m -- "$1")"
if [[ "${TARGET}" != /* || "${TARGET}" == "/" ]]; then
    echo "Refusing unsafe target path: ${TARGET}" >&2
    exit 1
fi

# A completed target is itself a resumable terminal state.
if [[ -f "${TARGET}/PREPARED_FROM_KAGGLE.txt" && \
      -d "${TARGET}/train" && -d "${TARGET}/val" ]]; then
    echo "ImageNet is already prepared; nothing to do: ${TARGET}"
    exit 0
fi
if [[ -e "${TARGET}" ]]; then
    echo "Target already exists but is not marked complete: ${TARGET}" >&2
    echo "Refusing to overwrite it." >&2
    exit 1
fi

TARGET_PARENT="$(dirname -- "${TARGET}")"
mkdir -p -- "${TARGET_PARENT}"

WORK_DIR="${TARGET}.download_tmp"
DOWNLOAD_DIR="${WORK_DIR}/download"
EXTRACT_DIR="${WORK_DIR}/direct_extract"
MARKER_DIR="${WORK_DIR}/markers"
PREPARED_DIR="${WORK_DIR}/prepared"
mkdir -p -- "${DOWNLOAD_DIR}" "${EXTRACT_DIR}" "${MARKER_DIR}"

# Prevent two instances of this resumable implementation from writing the
# same staging tree. Ctrl-C invokes the EXIT trap and releases this lock.
LOCK_DIR="${WORK_DIR}/.download_imagenet.lock"
if ! mkdir -- "${LOCK_DIR}" 2>/dev/null; then
    echo "Another download_imagenet.sh instance appears to be using: ${WORK_DIR}" >&2
    echo "If none is running, remove this stale lock directory: ${LOCK_DIR}" >&2
    exit 1
fi
cleanup_lock() {
    rmdir -- "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [[ -d "${WORK_DIR}/raw" ]]; then
    echo "Ignoring the legacy partial raw/ extraction; its completeness cannot be trusted."
fi

TRAIN_SOURCE="${EXTRACT_DIR}/ILSVRC/Data/CLS-LOC/train"
VAL_SOURCE="${EXTRACT_DIR}/ILSVRC/Data/CLS-LOC/val"
SYNSET_MAPPING="${EXTRACT_DIR}/LOC_synset_mapping.txt"
VAL_SOLUTION="${EXTRACT_DIR}/LOC_val_solution.csv"
EXTRACTION_COMPLETE="${MARKER_DIR}/direct_extraction.done"

extraction_ready=1
for required_path in \
    "${TRAIN_SOURCE}" "${VAL_SOURCE}" "${SYNSET_MAPPING}" "${VAL_SOLUTION}"; do
    if [[ ! -e "${required_path}" ]]; then
        extraction_ready=0
    fi
done
if [[ ! -f "${EXTRACTION_COMPLETE}" ]]; then
    extraction_ready=0
fi

COMPETITION="imagenet-object-localization-challenge"
ARCHIVE="${DOWNLOAD_DIR}/${COMPETITION}.zip"

if [[ "${extraction_ready}" != "1" ]]; then
    # If the 166.5 GB ZIP already exists, only space for selected extracted
    # train/val data is still needed. A fresh run needs space for both.
    if [[ -f "${ARCHIVE}" ]]; then
        DEFAULT_MIN_FREE_GIB=200
    else
        DEFAULT_MIN_FREE_GIB=350
    fi
    MIN_FREE_GIB="${MIN_FREE_GIB:-${DEFAULT_MIN_FREE_GIB}}"
    if [[ ! "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
        echo "MIN_FREE_GIB must be a non-negative integer: ${MIN_FREE_GIB}" >&2
        exit 1
    fi
    AVAILABLE_KIB="$(df -Pk -- "${TARGET_PARENT}" | awk 'NR==2 {print $4}')"
    REQUIRED_KIB="$((MIN_FREE_GIB * 1024 * 1024))"
    if [[ "${SKIP_SPACE_CHECK:-0}" != "1" && \
          "${AVAILABLE_KIB}" -lt "${REQUIRED_KIB}" ]]; then
        AVAILABLE_GIB="$((AVAILABLE_KIB / 1024 / 1024))"
        echo "Insufficient free space: ${AVAILABLE_GIB} GiB available, ${MIN_FREE_GIB} GiB required." >&2
        echo "Set SKIP_SPACE_CHECK=1 only after verifying storage capacity." >&2
        exit 1
    fi
fi

if [[ "${extraction_ready}" == "1" ]]; then
    echo "Skipping extraction: all direct-ZIP extraction stages are complete."
else
    if [[ -f "${ARCHIVE}" ]]; then
        echo "Reusing the existing Kaggle archive: ${ARCHIVE}"
    else
        if ! command -v kaggle >/dev/null 2>&1; then
            echo "Kaggle CLI is required because the archive is not present." >&2
            echo "Activate the project Conda environment or install kaggle first." >&2
            exit 1
        fi
        echo "Downloading ImageNet from Kaggle..."
        echo "This requires accepted competition rules and configured Kaggle credentials."
        kaggle competitions download -c "${COMPETITION}" -p "${DOWNLOAD_DIR}"
    fi
    if [[ ! -f "${ARCHIVE}" ]]; then
        echo "Kaggle did not create the expected archive: ${ARCHIVE}" >&2
        exit 1
    fi

    METADATA_MARKER="${MARKER_DIR}/metadata.done"
    if [[ -f "${METADATA_MARKER}" && \
          -s "${SYNSET_MAPPING}" && -s "${VAL_SOLUTION}" ]]; then
        echo "Skipping metadata extraction: already complete."
    else
        echo "Extracting ImageNet class and validation metadata..."
        unzip -q -o "${ARCHIVE}" \
            'LOC_synset_mapping.txt' 'LOC_val_solution.csv' \
            -d "${EXTRACT_DIR}"
        if [[ ! -s "${SYNSET_MAPPING}" || ! -s "${VAL_SOLUTION}" ]]; then
            echo "The ZIP does not use the expected direct Kaggle layout." >&2
            exit 1
        fi
        touch -- "${METADATA_MARKER}"
    fi

    mapfile -t WNIDS < <(awk 'NF {print $1}' "${SYNSET_MAPPING}")
    if [[ "${#WNIDS[@]}" -ne 1000 ]]; then
        echo "Expected 1000 class IDs in ${SYNSET_MAPPING}; got ${#WNIDS[@]}." >&2
        exit 1
    fi

    TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-50}"
    if [[ ! "${TRAIN_GROUP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_GROUP_SIZE must be a positive integer: ${TRAIN_GROUP_SIZE}" >&2
        exit 1
    fi

    total_classes="${#WNIDS[@]}"
    for ((group_start = 0; group_start < total_classes; group_start += TRAIN_GROUP_SIZE)); do
        group_end=$((group_start + TRAIN_GROUP_SIZE - 1))
        if ((group_end >= total_classes)); then
            group_end=$((total_classes - 1))
        fi
        group_marker="${MARKER_DIR}/train_$(printf '%04d_%04d' "${group_start}" "${group_end}").done"

        group_valid=1
        for ((index = group_start; index <= group_end; index++)); do
            if [[ ! -d "${TRAIN_SOURCE}/${WNIDS[$index]}" ]]; then
                group_valid=0
                break
            fi
        done
        if [[ -f "${group_marker}" && "${group_valid}" == "1" ]]; then
            echo "Skipping training classes $((group_start + 1))-$((group_end + 1))/${total_classes}: complete."
            continue
        fi

        echo "Extracting training classes $((group_start + 1))-$((group_end + 1))/${total_classes}..."
        patterns=()
        for ((index = group_start; index <= group_end; index++)); do
            patterns+=("ILSVRC/Data/CLS-LOC/train/${WNIDS[$index]}/*")
        done
        # -o is intentional: after an interruption, the unfinished group is
        # rewritten so a truncated JPEG cannot be mistaken for completed work.
        unzip -q -o "${ARCHIVE}" "${patterns[@]}" -d "${EXTRACT_DIR}"
        for ((index = group_start; index <= group_end; index++)); do
            if [[ ! -d "${TRAIN_SOURCE}/${WNIDS[$index]}" ]]; then
                echo "Training class was not extracted: ${WNIDS[$index]}" >&2
                exit 1
            fi
        done
        touch -- "${group_marker}"
    done

    VAL_MARKER="${MARKER_DIR}/validation.done"
    if [[ -f "${VAL_MARKER}" && -d "${VAL_SOURCE}" ]]; then
        echo "Skipping validation extraction: already complete."
    else
        echo "Extracting 50,000 validation images..."
        unzip -q -o "${ARCHIVE}" \
            'ILSVRC/Data/CLS-LOC/val/*' \
            -d "${EXTRACT_DIR}"
        if [[ ! -d "${VAL_SOURCE}" ]]; then
            echo "Validation directory was not extracted: ${VAL_SOURCE}" >&2
            exit 1
        fi
        touch -- "${VAL_MARKER}"
    fi

    touch -- "${EXTRACTION_COMPLETE}"
fi

for required_path in \
    "${TRAIN_SOURCE}" "${VAL_SOURCE}" "${SYNSET_MAPPING}" "${VAL_SOLUTION}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required extracted path is missing: ${required_path}" >&2
        exit 1
    fi
done

echo "Preparing resumable numeric train/validation directories..."
python3 - "${TRAIN_SOURCE}" "${VAL_SOURCE}" \
    "${SYNSET_MAPPING}" "${VAL_SOLUTION}" "${PREPARED_DIR}" <<'PY'
import csv
import os
from pathlib import Path

train_source = Path(os.sys.argv[1])
val_source = Path(os.sys.argv[2])
mapping_path = Path(os.sys.argv[3])
val_solution_path = Path(os.sys.argv[4])
prepared = Path(os.sys.argv[5])

wnids = []
with mapping_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            wnids.append(line.split(maxsplit=1)[0])
if len(wnids) != 1000 or len(set(wnids)) != 1000:
    raise RuntimeError(
        f"Expected 1000 unique synsets in {mapping_path}, got {len(wnids)}"
    )
wnid_to_index = {wnid: index for index, wnid in enumerate(wnids)}

prepared.mkdir(parents=True, exist_ok=True)
train_target = prepared / "train"
val_target = prepared / "val"
train_target.mkdir(exist_ok=True)
val_target.mkdir(exist_ok=True)
for index in range(1000):
    (val_target / str(index)).mkdir(exist_ok=True)

expected_wnids = set(wnids)
source_classes = {path.name for path in train_source.iterdir() if path.is_dir()}
extra_source = sorted(source_classes - expected_wnids)
expected_numeric = {str(index) for index in range(1000)}
target_classes = {path.name for path in train_target.iterdir() if path.is_dir()}
extra_target = sorted(target_classes - expected_numeric)
if extra_source or extra_target:
    raise RuntimeError(
        "Unexpected training directories: "
        f"source={extra_source[:10]}, prepared={extra_target[:10]}"
    )

# Source and destination are on the same filesystem. Each directory rename is
# atomic and cheap, and the source/destination state makes this loop resumable.
for completed, (wnid, index) in enumerate(wnid_to_index.items(), start=1):
    source = train_source / wnid
    destination = train_target / str(index)
    if source.is_dir() and not destination.exists():
        source.rename(destination)
    elif destination.is_dir() and not source.exists():
        pass
    elif source.exists() and destination.exists():
        raise RuntimeError(f"Both training paths exist for {wnid}")
    else:
        raise FileNotFoundError(
            f"Missing both training source and destination for {wnid}"
        )
    if completed % 100 == 0:
        print(f"Prepared {completed}/1000 training classes", flush=True)

validation_count = 0
seen_validation_ids = set()
with val_solution_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    required_columns = {"ImageId", "PredictionString"}
    if not required_columns.issubset(reader.fieldnames or []):
        raise RuntimeError(
            f"Unexpected columns in {val_solution_path}: {reader.fieldnames}"
        )
    for row in reader:
        image_id = row["ImageId"]
        if image_id in seen_validation_ids:
            raise RuntimeError(f"Duplicate validation image ID: {image_id}")
        seen_validation_ids.add(image_id)
        wnid = row["PredictionString"].split()[0]
        if wnid not in wnid_to_index:
            raise RuntimeError(f"Unknown validation synset {wnid} for {image_id}")

        source = val_source / f"{image_id}.JPEG"
        destination = val_target / str(wnid_to_index[wnid]) / source.name
        if source.is_file() and not destination.exists():
            source.rename(destination)
        elif destination.is_file() and not source.exists():
            pass
        elif source.exists() and destination.exists():
            raise RuntimeError(f"Both validation paths exist for {image_id}")
        else:
            raise FileNotFoundError(
                f"Missing both validation source and destination for {image_id}"
            )

        validation_count += 1
        if validation_count % 5000 == 0:
            print(f"Prepared {validation_count}/50000 validation images", flush=True)

if validation_count != 50_000:
    raise RuntimeError(
        f"Expected 50,000 validation rows, got {validation_count}"
    )

image_suffixes = {".jpeg", ".jpg", ".png"}
print("Validating final ImageNet file counts...", flush=True)
train_count = sum(
    path.suffix.lower() in image_suffixes for path in train_target.rglob("*")
)
val_count = sum(
    path.suffix.lower() in image_suffixes for path in val_target.rglob("*")
)
nonempty_train_classes = sum(any(path.iterdir()) for path in train_target.iterdir())
nonempty_val_classes = sum(any(path.iterdir()) for path in val_target.iterdir())
if nonempty_train_classes != 1000 or nonempty_val_classes != 1000:
    raise RuntimeError(
        "Prepared data has empty classes: "
        f"train_nonempty={nonempty_train_classes}, "
        f"val_nonempty={nonempty_val_classes}"
    )
if train_count != 1_281_167:
    raise RuntimeError(f"Expected 1,281,167 training images, got {train_count}")
if val_count != 50_000:
    raise RuntimeError(f"Expected 50,000 validation images, got {val_count}")

marker = prepared / "PREPARED_FROM_KAGGLE.txt"
temporary_marker = prepared / "PREPARED_FROM_KAGGLE.txt.tmp"
temporary_marker.write_text(
    "source=imagenet-object-localization-challenge\n"
    f"train_images={train_count}\n"
    f"val_images={val_count}\n"
    "class_mapping=zero-based LOC_synset_mapping.txt line number\n",
    encoding="utf-8",
)
os.replace(temporary_marker, marker)
print(f"Prepared {train_count} train images and {val_count} validation images")
PY

mv -- "${PREPARED_DIR}" "${TARGET}"

if [[ "${KEEP_DOWNLOAD:-0}" == "1" ]]; then
    echo "Keeping download and extraction metadata in ${WORK_DIR}"
else
    # WORK_DIR is a validated, dedicated sibling with a fixed suffix. TARGET
    # has already been atomically moved out of it.
    rm -rf -- "${WORK_DIR}"
fi

echo "ImageNet is ready at: ${TARGET}"
echo "Use: DATA_ROOT=${TARGET}"
