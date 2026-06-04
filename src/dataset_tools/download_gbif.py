"""Download a GBIF Darwin Core Archive filtered by verbatim scientific names."""

import os
import time

from dotenv import load_dotenv

load_dotenv()

INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
TERMINAL_STATUSES = {"SUCCEEDED", "CANCELLED", "KILLED", "FAILED", "SUSPENDED", "FILE_ERASED"}


def read_names_from_file(names_file: str) -> list[str]:
    """Read one name per line; skip blank lines and #-prefixed comments."""
    names = []
    with open(names_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def bbox_to_wkt(bbox: str) -> str:
    """Convert 'min_lat,min_lng,max_lat,max_lng' to a WKT POLYGON string.

    GBIF requires lng,lat order in WKT coordinates.
    """
    parts = bbox.split(",")
    if len(parts) != 4:
        raise ValueError(f"bbox must be 4 comma-separated floats, got: {bbox!r}")
    try:
        min_lat, min_lng, max_lat, max_lng = [float(p.strip()) for p in parts]
    except ValueError:
        raise ValueError(f"bbox values must be floats, got: {bbox!r}")
    return (
        f"POLYGON(("
        f"{min_lng} {min_lat}, "
        f"{max_lng} {min_lat}, "
        f"{max_lng} {max_lat}, "
        f"{min_lng} {max_lat}, "
        f"{min_lng} {min_lat}"
        f"))"
    )


def build_predicate(
    names: list[str],
    dataset_key: str = INAT_DATASET_KEY,
    countries: list[str] | None = None,
    bbox: str | None = None,
) -> dict:
    """Build a GBIF download predicate dict from the given parameters."""
    if not names:
        raise ValueError("names must not be empty")

    predicates = []

    if len(names) == 1:
        predicates.append(
            {"type": "equals", "key": "VERBATIM_SCIENTIFIC_NAME", "value": names[0]}
        )
    else:
        predicates.append(
            {"type": "in", "key": "VERBATIM_SCIENTIFIC_NAME", "values": names}
        )

    predicates.append({"type": "equals", "key": "DATASET_KEY", "value": dataset_key})
    predicates.append({"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"})

    if countries:
        if len(countries) == 1:
            predicates.append({"type": "equals", "key": "COUNTRY", "value": countries[0]})
        else:
            predicates.append({"type": "in", "key": "COUNTRY", "values": list(countries)})

    if bbox:
        predicates.append({"type": "within", "geometry": bbox_to_wkt(bbox)})

    return {"type": "and", "predicates": predicates}


def submit_download(predicate: dict) -> str:
    """Submit a GBIF download request and return the download key."""
    gbif_user = os.environ.get("GBIF_USER")
    gbif_pwd = os.environ.get("GBIF_PWD")
    gbif_email = os.environ.get("GBIF_EMAIL")

    missing = [name for name, val in [("GBIF_USER", gbif_user), ("GBIF_PWD", gbif_pwd), ("GBIF_EMAIL", gbif_email)] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file or environment before running."
        )

    from pygbif import occurrences as occ

    res = occ.download(predicate, format="DWCA", user=gbif_user, pwd=gbif_pwd, email=gbif_email)
    return res[0]


def poll_download(key: str, poll_interval: int = 30, max_wait: int = 3600) -> dict:
    """Poll GBIF until the download reaches a terminal status.

    Returns the metadata dict on SUCCEEDED; raises on failure or timeout.
    """
    from pygbif import occurrences as occ

    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        meta = occ.download_meta(key)
        status = meta.get("status", "UNKNOWN")
        print(f"[{elapsed}s] status: {status}")

        if status in TERMINAL_STATUSES:
            if status == "SUCCEEDED":
                return meta
            raise RuntimeError(
                f"GBIF download {key} ended with non-success status: {status}"
            )

        if elapsed >= max_wait:
            raise TimeoutError(
                f"GBIF download {key} did not complete within {max_wait}s (last status: {status})"
            )

        time.sleep(poll_interval)


def fetch_archive(key: str, output_dir: str) -> str:
    """Download the completed GBIF archive zip to output_dir and return its path."""
    from pygbif import occurrences as occ

    os.makedirs(output_dir, exist_ok=True)
    occ.download_get(key, path=output_dir)
    zip_path = os.path.join(output_dir, f"{key}.zip")
    return zip_path


def download_gbif(
    names: list[str],
    output_dir: str,
    dataset_key: str = INAT_DATASET_KEY,
    countries: list[str] | None = None,
    bbox: str | None = None,
    poll_interval: int = 30,
    max_wait: int = 3600,
) -> str:
    """Orchestrate a full GBIF DwC-A download filtered by verbatim scientific names.

    Returns the path to the downloaded zip file.
    """
    names = list(dict.fromkeys(names))  # deduplicate, preserve order
    print(f"Requesting GBIF download for {len(names)} unique name(s).")

    predicate = build_predicate(names, dataset_key=dataset_key, countries=countries, bbox=bbox)
    key = submit_download(predicate)
    print(f"Download submitted. Key: {key}")

    poll_download(key, poll_interval=poll_interval, max_wait=max_wait)

    zip_path = fetch_archive(key, output_dir)
    print(f"Archive saved to: {zip_path}")
    print(f"Hint: ami-dataset fetch-images --dwca-file {zip_path} --dataset-path <output_images_dir>")
    return zip_path
