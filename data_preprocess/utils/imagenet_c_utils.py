import tarfile
import urllib.request
from datasets import Dataset, DatasetDict, Image as HFImage
from tqdm import tqdm
import shutil

IMAGENET_C_BASE_URL = "https://zenodo.org/record/2235448/files"
IMAGENET_C_ARCHIVES = ["blur.tar", "digital.tar", "noise.tar", "weather.tar"]


def download_file(url, dest_path):
    """
    Downloads a file from url to dest_path with a tqdm progress bar.
    Skips the download if the destination file already exists.
    """
    if dest_path.exists():
        logger.info(f"{dest_path.name} already present, skipping download")
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024 * 1024  # 1 MB

            with open(tmp_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {dest_path.name}",
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    pbar.update(len(chunk))

        tmp_path.rename(dest_path)

    except Exception:
        # Remove incomplete download so it can be retried cleanly
        if tmp_path.exists():
            tmp_path.unlink()
        raise

def download_and_extract_imagenet_c(original_save_folder):
    """
    Downloads the four official ImageNet-C archives from Zenodo and extracts
    only the classes present in `wnids` into original_save_folder/imagenet-c.
    Returns the path to the extracted folder.
    """
    archive_download_folder = original_save_folder / "imagenet-c-archives"
    archive_download_folder.mkdir(exist_ok=True)

    extracted_folder = original_save_folder / "imagenet-c"
    extracted_folder.mkdir(exist_ok=True)

    for archive_name in IMAGENET_C_ARCHIVES:
        archive_path = archive_download_folder / archive_name
        download_file(f"{IMAGENET_C_BASE_URL}/{archive_name}", archive_path)
        with tarfile.open(archive_path, "r") as tar:
            members = tar.getmembers()
            for member in tqdm(members,desc=f"Extracting {archive_name}",unit="files",):
                tar.extract(member, path=extracted_folder)
    shutil.rmtree(archive_download_folder)
    return extracted_folder


def build_imagenet_c_datasets(extracted_folder, wnid_to_idx):
    """
    Walks the extracted ImageNet-C folder (structure: corruption/severity/wnid/*.JPEG)
    and builds a SEPARATE HuggingFace Dataset for every (corruption, severity) pair,
    e.g. "gaussian_noise_sev_1", "gaussian_noise_sev_2", ..., "fog_sev_5".
 
    Each individual dataset has columns: image, wnid, label
    (corruption and severity are implicit in the dataset's key, so they're
    dropped from the per-row records to avoid redundant columns).
 
    Returns a datasets.DatasetDict mapping "<corruption>_sev_<severity>" -> Dataset.
    """
    datasets_by_key = {}
 
    for corruption_dir in sorted(extracted_folder.iterdir()):
        if not corruption_dir.is_dir():
            continue
 
        for severity_dir in sorted(corruption_dir.iterdir()):
            severity = int(severity_dir.name)
            records = []
 
            for wnid_dir in sorted(severity_dir.iterdir()):
                wnid = wnid_dir.name
                if wnid not in wnid_to_idx:
                    continue
                for image_path in wnid_dir.iterdir():
                    records.append({
                        "image": str(image_path),
                        "wnid": wnid,
                        "label": wnid_to_idx[wnid],
                    })
 
            key = f"{corruption_dir.name}_sev_{severity}"
 
            if not records:
                # no images matched wnid_to_idx for this corruption/severity
                continue
 
            dataset = Dataset.from_list(records)
            dataset = dataset.cast_column("image", HFImage())
            datasets_by_key[key] = dataset
 
    return DatasetDict(datasets_by_key)