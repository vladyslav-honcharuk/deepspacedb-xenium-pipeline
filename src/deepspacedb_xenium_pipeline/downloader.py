"""Pipeline stage 2: download Xenium sample files from NCBI GEO.

Parses each sample's MiniML metadata to discover supplementary file URLs and
streams them into ``{data_dir}/{GPL}/{GSE}/{GSM}/raw/``, with resume support via
a ``.download_complete`` marker. Refactored from the original
``xenium_downloader.py``: shared :class:`PipelineConfig`, package logger (no
``logging.basicConfig`` on the root logger), no directory creation as an import/
construction side effect, and the shared :class:`DownloadResult` type.
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import PipelineConfig
from .exceptions import DownloadError
from .logging_setup import get_logger
from .results import DownloadResult

_MINIML_URL = "https://www.ncbi.nlm.nih.gov/geo/tools/geometa.cgi"
_MINIML_NS = "{http://www.ncbi.nlm.nih.gov/geo/info/MINiML}"


class _DownloadInfo:
    """Internal container for URLs/ids extracted from a sample's MiniML."""

    __slots__ = ("gsm_id", "supplementary_files", "miniml_url", "gpl_id", "gse_id")

    def __init__(self, gsm_id, supplementary_files, miniml_url, gpl_id, gse_id):
        self.gsm_id = gsm_id
        self.supplementary_files = supplementary_files
        self.miniml_url = miniml_url
        self.gpl_id = gpl_id
        self.gse_id = gse_id


class SampleDownloader:
    """Download Xenium GSM samples from NCBI GEO with resume support."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        self.session = self._build_session()
        self._sample_metadata: Optional[pd.DataFrame] = None

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.config.request_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def load_sample_metadata(self, csv_path: Path) -> None:
        """Load a ``platform, series, sample`` CSV for GPL/GSE lookups."""
        try:
            df = pd.read_csv(csv_path)
            missing = [c for c in ("platform", "series", "sample") if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            self._sample_metadata = df.set_index("sample")
            self.logger.info("Loaded metadata for %d samples from %s", len(self._sample_metadata), csv_path)
        except Exception as exc:  # noqa: BLE001 - metadata is optional context
            self.logger.error("Failed to load sample metadata: %s", exc)
            self._sample_metadata = None

    def load_samples_from_csv(self, csv_path: Path) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """Return ``(gsm, gpl, gse)`` tuples from a samples CSV."""
        df = pd.read_csv(csv_path)
        missing = [c for c in ("platform", "series", "sample") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        samples = [(row["sample"], row["platform"], row["series"]) for _, row in df.iterrows()]
        self.logger.info("Loaded %d samples from %s", len(samples), csv_path)
        return samples

    def _lookup_metadata(self, gsm_id: str) -> Tuple[Optional[str], Optional[str]]:
        if self._sample_metadata is None or gsm_id not in self._sample_metadata.index:
            return None, None
        row = self._sample_metadata.loc[gsm_id]
        return row["platform"], row["series"]

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def download_gsm_sample(
        self, gsm_id: str, gpl_id: Optional[str] = None, series_id: Optional[str] = None
    ) -> DownloadResult:
        """Download one GSM sample, resuming if already complete."""
        self.logger.info("Downloading GSM sample: %s", gsm_id)
        try:
            if not gpl_id or not series_id:
                meta_gpl, meta_gse = self._lookup_metadata(gsm_id)
                gpl_id = gpl_id or meta_gpl
                series_id = series_id or meta_gse

            info = self.get_download_info(gsm_id)
            if info is None:
                raise DownloadError(f"Failed to get download info for {gsm_id}")

            gpl_id = gpl_id or info.gpl_id or "unknown_platform"
            series_id = series_id or info.gse_id or "unknown_series"
            self.logger.info("Using GPL=%s GSE=%s", gpl_id, series_id)

            sample_dir = self.config.data_dir / gpl_id / series_id / gsm_id
            if self._is_downloaded(sample_dir):
                self.logger.info("%s already downloaded; skipping", gsm_id)
                return DownloadResult(
                    sample_path=sample_dir, success=True, gsm_id=gsm_id, raw_path=sample_dir / "raw"
                )

            downloaded = self._download_from_info(info, sample_dir)
            result = DownloadResult(
                sample_path=sample_dir, success=True, gsm_id=gsm_id,
                raw_path=sample_dir / "raw", files_processed=[f.name for f in downloaded],
            )
            self._write_completion_marker(sample_dir, result)
            self.logger.info("Downloaded %s (%d files)", gsm_id, len(downloaded))
            return result
        except Exception as exc:  # noqa: BLE001 - boundary: convert to result
            msg = f"Failed to download {gsm_id}: {exc}"
            self.logger.error(msg)
            return DownloadResult(sample_path=self.config.data_dir, success=False, gsm_id=gsm_id, error_message=msg)

    def download_multiple(
        self, samples: List[Tuple[str, Optional[str], Optional[str]]]
    ) -> Dict[str, DownloadResult]:
        """Download a batch of ``(gsm, gpl, gse)`` tuples."""
        results: Dict[str, DownloadResult] = {}
        total = len(samples)
        self.logger.info("Starting batch download of %d samples", total)
        for i, (gsm_id, gpl_id, series_id) in enumerate(samples, 1):
            self.logger.info("Sample %d/%d: %s", i, total, gsm_id)
            results[gsm_id] = self.download_gsm_sample(gsm_id, gpl_id, series_id)
            if i < total:
                time.sleep(self.config.download_delay)
        successful = sum(1 for r in results.values() if r.success)
        self.logger.info("Batch download complete: %d/%d successful", successful, total)
        failed = [g for g, r in results.items() if not r.success]
        if failed:
            self.logger.warning("Failed samples: %s", ", ".join(failed))
        return results

    def discover_downloaded_samples(self, base_dir: Optional[Path] = None) -> List[Path]:
        """Return sample dirs with a completed download under ``base_dir``."""
        import glob

        search_dir = base_dir or self.config.data_dir
        found = []
        for raw_dir in glob.glob(str(search_dir / "**/raw/"), recursive=True):
            sample_dir = Path(raw_dir).parent
            if self._is_downloaded(sample_dir):
                found.append(sample_dir)
        self.logger.info("Found %d downloaded samples", len(found))
        return found

    # ------------------------------------------------------------------ #
    # NCBI MiniML
    # ------------------------------------------------------------------ #

    def get_download_info(self, gsm_id: str) -> Optional[_DownloadInfo]:
        self.logger.info("Fetching download info for %s", gsm_id)
        params = {"acc": gsm_id, "scope": "full", "mode": "miniml"}
        try:
            response = self.session.get(_MINIML_URL, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError) as exc:
            self.logger.error("Failed to get download info for %s: %s", gsm_id, exc)
            return None

        supplementary = [
            sd.text.strip()
            for sd in root.findall(f".//{_MINIML_NS}Supplementary-Data")
            if sd.text and (sd.text.strip().startswith("http") or sd.text.strip().startswith("ftp"))
        ]
        if not supplementary:
            self.logger.warning("No supplementary files for %s", gsm_id)
            return None

        gpl_id = gse_id = None
        platform_ref = root.find(f".//{_MINIML_NS}Platform-Ref")
        if platform_ref is not None:
            gpl_id = platform_ref.get("ref")
        series_ref = root.find(f".//{_MINIML_NS}Series-Ref")
        if series_ref is not None:
            gse_id = series_ref.get("ref")
        if not gpl_id or not gse_id:
            for relation in root.findall(f".//{_MINIML_NS}Relation"):
                rtype = relation.get("type", "").lower()
                target = relation.get("target", "")
                if "platform" in rtype and target.startswith("GPL"):
                    gpl_id = gpl_id or target
                elif "series" in rtype and target.startswith("GSE"):
                    gse_id = gse_id or target

        return _DownloadInfo(
            gsm_id=gsm_id,
            supplementary_files=supplementary,
            miniml_url=f"{_MINIML_URL}?{requests.compat.urlencode(params)}",
            gpl_id=gpl_id,
            gse_id=gse_id,
        )

    def _download_from_info(self, info: _DownloadInfo, sample_dir: Path) -> List[Path]:
        sample_dir.mkdir(parents=True, exist_ok=True)
        miniml_path = sample_dir / f"{info.gsm_id}_miniml.xml"
        if not miniml_path.exists():
            try:
                response = self.session.get(info.miniml_url, timeout=self.config.request_timeout)
                response.raise_for_status()
                miniml_path.write_bytes(response.content)
                self.logger.info("Saved MiniML XML to %s", miniml_path.name)
            except requests.RequestException as exc:
                self.logger.warning("Failed to save MiniML XML: %s", exc)

        raw_dir = sample_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []
        for url in info.supplementary_files:
            filename = Path(urlparse(url).path).name or f"file_{len(downloaded)}"
            file_path = raw_dir / filename
            if self._download_file(url, file_path):
                downloaded.append(file_path)
            time.sleep(self.config.download_delay)
        if not downloaded:
            raise DownloadError(f"No files downloaded for {info.gsm_id}")
        return downloaded

    def _download_file(self, url: str, output_path: Path) -> bool:
        try:
            self.logger.info("Downloading %s -> %s", url, output_path.name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session.get(url, stream=True, timeout=self.config.request_timeout) as response:
                response.raise_for_status()
                with open(output_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
            self.logger.info("Downloaded %s", output_path.name)
            return True
        except requests.RequestException as exc:
            self.logger.error("Failed to download %s: %s", url, exc)
            output_path.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------ #
    # Resume markers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_downloaded(sample_dir: Path) -> bool:
        if not (sample_dir.exists() and (sample_dir / ".download_complete").exists()):
            return False
        raw_dir = sample_dir / "raw"
        return raw_dir.exists() and any(raw_dir.iterdir())

    @staticmethod
    def _write_completion_marker(sample_dir: Path, result: DownloadResult) -> None:
        marker = sample_dir / ".download_complete"
        marker.write_text(
            json.dumps(
                {
                    "gsm_id": result.gsm_id,
                    "completion_time": time.time(),
                    "files_downloaded": result.files_processed,
                    "success": result.success,
                },
                indent=2,
            )
        )
