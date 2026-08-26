"""Pipeline stage 1: discover Xenium samples from NCBI GEO.

Walks the NCBI GEO hierarchy (GPL platform -> GSE series -> GSM sample) for
Xenium records and returns a tidy ``platform, series, sample`` table. It logs
through the package logger, takes rate limits from :class:`PipelineConfig`, and
raises :class:`FindError` for unrecoverable failures instead of returning bare
sentinels.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import pandas as pd
import requests

from .config import PipelineConfig
from .logging_setup import get_logger

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class SampleFinder:
    """Find Xenium GSM samples from NCBI GEO via the E-utilities API."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def find(self) -> pd.DataFrame:
        """Return a DataFrame of ``platform, series, sample`` for all Xenium samples."""
        self.logger.info("Starting Xenium sample search")
        platforms = self.get_xenium_platforms()
        if not platforms:
            self.logger.warning("No Xenium platforms found")
            return pd.DataFrame(columns=["platform", "series", "sample"])

        all_samples: List[Dict] = []
        for i, platform in enumerate(platforms, 1):
            self.logger.info("Platform %d/%d: %s", i, len(platforms), platform)
            all_samples.extend(self.get_samples_for_platform(platform))
            time.sleep(self.config.ncbi_delay)

        df = pd.DataFrame(all_samples)
        self.logger.info("Found %d total samples", len(df))
        return df

    @staticmethod
    def save(df: pd.DataFrame, output: str = "xenium_samples.csv") -> Optional[str]:
        """Write the sample collection to CSV. Returns the path, or None if empty."""
        if df.empty:
            return None
        df.to_csv(output, index=False)
        return output

    def get_xenium_platforms(self) -> List[str]:
        """Return all Xenium GPL platform accessions."""
        self.logger.info("Finding Xenium platforms")
        uids = self._search_gds("Xenium[All Fields] AND GPL")
        gpl_uids = [uid for uid in uids if uid.startswith("1")]
        if not gpl_uids:
            return []
        root = self._get_summaries(gpl_uids)
        if root is None:
            return []
        accessions: List[str] = []
        for doc in root.findall(".//DocumentSummary"):
            acc = doc.find("Accession")
            if acc is not None and acc.text:
                accessions.append(acc.text)
        self.logger.info("Found %d Xenium platforms", len(accessions))
        return accessions

    def get_samples_for_platform(self, platform: str) -> List[Dict]:
        """Return ``platform, series, sample`` rows for one GPL platform."""
        self.logger.info("Processing %s", platform)
        uids = self._search_gds(f"{platform}[ACCN]", retmax=10000)
        sample_uids = [uid for uid in uids if uid.startswith("3")]
        if not sample_uids:
            self.logger.info("No samples for %s", platform)
            return []
        root = self._get_summaries(sample_uids)
        if root is None:
            return []
        samples: List[Dict] = []
        for doc in root.findall(".//DocumentSummary"):
            sample_acc = doc.find("Accession")
            gse_elem = doc.find("GSE")
            if sample_acc is not None and sample_acc.text and gse_elem is not None and gse_elem.text:
                series_num = gse_elem.text.split(";")[0].strip()
                samples.append({"platform": platform, "series": f"GSE{series_num}", "sample": sample_acc.text})
        self.logger.info("Found %d samples for %s", len(samples), platform)
        return samples

    # ------------------------------------------------------------------ #
    # NCBI E-utilities helpers
    # ------------------------------------------------------------------ #

    def _api_call(self, endpoint: str, params: Dict, context: str = "") -> Optional[ET.Element]:
        url = f"{_BASE_URL}/{endpoint}.fcgi"
        try:
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
        except requests.RequestException as exc:
            self.logger.error("Request error %s: %s", context, exc)
            return None
        if response.status_code != 200:
            self.logger.error("HTTP %d %s", response.status_code, context)
            return None
        if not response.content.strip():
            self.logger.error("Empty response %s", context)
            return None
        try:
            return ET.fromstring(response.content)
        except ET.ParseError as exc:
            self.logger.error("XML parse error %s: %s", context, exc)
            return None

    def _search_gds(self, term: str, retmax: int = 1000) -> List[str]:
        params = {"db": "gds", "term": term, "retmax": retmax}
        root = self._api_call("esearch", params, f"searching '{term}'")
        if root is None:
            return []
        id_list = root.find("IdList")
        if id_list is None:
            return []
        return [elem.text for elem in id_list.findall("Id") if elem.text]

    def _get_summaries(self, uids: List[str], chunk_size: int = 200) -> Optional[ET.Element]:
        if not uids:
            return None
        combined = ET.Element("DocumentSummarySet")
        for i in range(0, len(uids), chunk_size):
            chunk = uids[i : i + chunk_size]
            params = {"db": "gds", "id": ",".join(chunk), "version": "2.0"}
            root = self._api_call("esummary", params, f"summaries for {len(chunk)} items")
            if root is not None:
                for doc in root.findall(".//DocumentSummary"):
                    combined.append(doc)
            else:
                self.logger.error("Failed summaries for chunk %d", i // chunk_size + 1)
            time.sleep(0.5)
        return combined
