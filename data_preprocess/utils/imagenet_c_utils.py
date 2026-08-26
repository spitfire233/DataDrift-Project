import tarfile
import urllib.request
from datasets import Dataset, Image as HFImage
from tqdm import tqdm

IMAGENET_C_BASE_URL = "https://zenodo.org/record/2235448/files"
IMAGENET_C_ARCHIVES = ["blur.tar", "digital.tar", "noise.tar", "weather.tar"]


def download_file(url, dest_path):
    """
    Downloads a file from url to dest_path, skipping the download if the
    destination file already exists (allows resuming an interrupted pipeline
    without redownloading tens of GB).
    """
    if dest_path.exists():
        logger.info(f"{dest_path.name} already present, skipping download")
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.rename(dest_path)

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
    return extracted_folder


def build_imagenet_c_dataset(extracted_folder, wnid_to_idx):
    """
    Walks the extracted ImageNet-C folder (structure: corruption/severity/wnid/*.JPEG)
    and builds a HuggingFace Dataset with columns: image, wnid, label, corruption, severity.
    """
    records = []
    for corruption_dir in sorted(extracted_folder.iterdir()):
        if not corruption_dir.is_dir():
            continue
        for severity_dir in sorted(corruption_dir.iterdir()):
            severity = int(severity_dir.name)
            for wnid_dir in sorted(severity_dir.iterdir()):
                wnid = wnid_dir.name
                if wnid not in wnid_to_idx:
                    continue
                for image_path in wnid_dir.iterdir():
                    records.append({
                        "image": str(image_path),
                        "wnid": wnid,
                        "label": wnid_to_idx[wnid],
                        "corruption": corruption_dir.name,
                        "severity": severity,
                    })
    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("image", HFImage())
    return dataset