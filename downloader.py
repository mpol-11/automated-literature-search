import os
import re
import requests
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

UNPAYWALL_EMAIL = "your_email@example.com"  # optional but recommended


@dataclass
class DownloadResult:
    success: bool
    source_url: str
    local_path: Optional[str] = None
    fallback_url: Optional[str] = None
    reason: Optional[str] = None


# --------------------------------------------------
# Downloader
# --------------------------------------------------
class PaperDownloader:
    def __init__(self, out_dir="downloaded_papers"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })

    # ---------------- utilities ----------------

    def safe_filename(self, url):
        name = os.path.basename(urlparse(url).path)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name or "paper.pdf"

    def save_pdf(self, response, source_url):
        path = os.path.join(self.out_dir, self.safe_filename(source_url))
        with open(path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        return path

    def try_direct_pdf(self, url):
        try:
            r = self.session.get(url, stream=True, timeout=20, allow_redirects=True)
        except Exception:
            return None

        if "application/pdf" in r.headers.get("Content-Type", "").lower():
            return r

        return None

    # ---------------- site helpers ----------------

    def try_suffix_pdf(self, base_url, suffixes):
        for s in suffixes:
            pdf_url = base_url.rstrip("/") + s
            pdf = self.try_direct_pdf(pdf_url)
            if pdf:
                return pdf, pdf_url
        return None, None

    def resolve_entry_url(self, url):
        return url  # placeholder (kept exactly, now valid)

    def resolve_doi_redirect(self, url):
        r = self.session.get(url, allow_redirects=True, timeout=15)
        return r.url

    # ---------------- PubMed metadata ----------------

    def get_pubmed_xml(self, pubmed_url):
        pmid = pubmed_url.rstrip("/").split("/")[-1]
        api = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&retmode=xml"
        )
        r = requests.get(api, timeout=20)
        return r.text if r.status_code == 200 else None

    def get_doi_from_pubmed(self, pubmed_url):
        xml = self.get_pubmed_xml(pubmed_url)
        if not xml:
            return None
        soup = BeautifulSoup(xml, "xml")
        tag = soup.find("ArticleId", {"IdType": "doi"})
        return tag.text if tag else None

    def get_pmcid_from_pubmed(self, pubmed_url):
        xml = self.get_pubmed_xml(pubmed_url)
        if not xml:
            return None
        soup = BeautifulSoup(xml, "xml")
        tag = soup.find("ArticleId", {"IdType": "pmc"})
        return tag.text if tag else None

    # ---------------- Unpaywall ----------------

    def unpaywall_lookup(self, doi):
        api = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
        r = requests.get(api, timeout=20)
        if r.status_code != 200:
            return None

        data = r.json()
        locations = data.get("oa_locations", [])

        for loc in locations:
            if loc.get("host_type") == "publisher" and loc.get("url_for_pdf"):
                return loc["url_for_pdf"]

        for loc in locations:
            if loc.get("host_type") == "publisher" and loc.get("url"):
                return loc["url"]

        best = data.get("best_oa_location")
        if best:
            return best.get("url_for_pdf") or best.get("url")

        return None

    # ---------------- DOI → Publisher page ----------------
    def is_doi_url(self, url):
      return "doi.org/" in url or url.startswith("10.")

    def resolve_doi(self, doi):
        doi_url = f"https://doi.org/{doi}"
        try:
            r = self.session.get(doi_url, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                return r.url, r.text
        except Exception:
            pass
        return None, None

    def extract_pdf_from_html(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if ".pdf" in href.lower():
                if href.startswith("/"):
                    parsed = urlparse(base_url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                return href

        return None

    # ---------------- PMC ----------------

    def try_pmc_pdf(self, pmcid):
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        return self.try_direct_pdf(pdf_url), pdf_url

    # ---------------- handlers ----------------

    def handle_pubmed(self, url):
        print("🔬 PubMed handler")

        pmcid = self.get_pmcid_from_pubmed(url)
        if pmcid:
            pdf, pdf_url = self.try_pmc_pdf(pmcid)
            if pdf:
                path = self.save_pdf(pdf, pdf_url)
                return DownloadResult(True, url, local_path=path)

        doi = self.get_doi_from_pubmed(url)
        if not doi:
            return DownloadResult(False, url, reason="No PMC and no DOI")

        landing_url, html = self.resolve_doi(doi)
        if not landing_url or not html:
            return DownloadResult(
                False,
                url,
                fallback_url=f"https://doi.org/{doi}",
                reason="Could not resolve publisher page"
            )

        pdf_url = self.extract_pdf_from_html(html, landing_url)
        if pdf_url:
            pdf = self.try_direct_pdf(pdf_url)
            if pdf:
                path = self.save_pdf(pdf, pdf_url)
                return DownloadResult(True, url, local_path=path)

        return DownloadResult(
            False,
            url,
            fallback_url=landing_url,
            reason="Publisher page (manual download)"
        )

    def handle_arxiv(self, url):
        print("📄 arXiv handler")

        m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)", url)
        if not m:
            return DownloadResult(False, url, reason="Invalid arXiv URL")

        pdf_url = f"https://arxiv.org/pdf/{m.group(2)}.pdf"
        pdf = self.try_direct_pdf(pdf_url)

        if pdf:
            path = self.save_pdf(pdf, pdf_url)
            return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, reason="arXiv PDF failed")

    def handle_generic(self, url):
        print("🌐 Generic handler")

        pdf = self.try_direct_pdf(url)
        if pdf:
            path = self.save_pdf(pdf, url)
            return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, reason="No direct PDF")

    def handle_biorxiv(self, url):
        print("🧬 bioRxiv / medRxiv handler")

        url = url.split("?")[0]

        pdf, pdf_url = self.try_suffix_pdf(
            url,
            [".full.pdf", "/full.pdf"]
        )

        if pdf:
            path = self.save_pdf(pdf, pdf_url)
            return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, reason="bioRxiv PDF not found")

    def handle_nature(self, url):
        print("🌿 Nature handler")

        r = self.session.get(url, timeout=20)
        if r.status_code != 200:
            return DownloadResult(False, url, reason="Nature page unreachable")

        pdf_url = self.extract_pdf_from_html(r.text, r.url)
        if pdf_url:
            pdf = self.try_direct_pdf(pdf_url)
            if pdf:
                path = self.save_pdf(pdf, pdf_url)
                return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, fallback_url=url, reason="Nature PDF not found")

    def handle_science(self, url):
        print("🔬 Science handler")

        r = self.session.get(url, timeout=20)
        if r.status_code != 200:
            return DownloadResult(False, url, reason="Science page unreachable")

        pdf_url = self.extract_pdf_from_html(r.text, r.url)
        if pdf_url:
            pdf = self.try_direct_pdf(pdf_url)
            if pdf:
                path = self.save_pdf(pdf, pdf_url)
                return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, fallback_url=url, reason="Science PDF not found")

    def handle_cell(self, url):
        print("🧫 Cell Press handler")

        r = self.session.get(url, timeout=20)
        if r.status_code != 200:
            return DownloadResult(False, url, reason="Cell page unreachable")

        pdf_url = self.extract_pdf_from_html(r.text, r.url)
        if pdf_url:
            pdf = self.try_direct_pdf(pdf_url)
            if pdf:
                path = self.save_pdf(pdf, pdf_url)
                return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, fallback_url=url, reason="Cell PDF not found")

    def handle_frontiers(self, url):
        print("🌍 Frontiers handler")

        pdf, pdf_url = self.try_suffix_pdf(url, ["/pdf"])
        if pdf:
            path = self.save_pdf(pdf, pdf_url)
            return DownloadResult(True, url, local_path=path)

        r = self.session.get(url, timeout=20)
        if r.status_code == 200:
            pdf_url = self.extract_pdf_from_html(r.text, r.url)
            if pdf_url:
                pdf = self.try_direct_pdf(pdf_url)
                if pdf:
                    path = self.save_pdf(pdf, pdf_url)
                    return DownloadResult(True, url, local_path=path)

        return DownloadResult(False, url, fallback_url=url, reason="Frontiers PDF not found")

    # ---------------- router ----------------
    def get_handler(self, url):
      resolved_url = url

    # 🔁 Resolve DOI first
      if self.is_doi_url(url):
        print("🔁 Resolving DOI...")
        try:
            r = self.session.get(url, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                resolved_url = r.url
        except Exception:
            pass

      domain = urlparse(resolved_url).netloc.lower()

      if "pubmed.ncbi.nlm.nih.gov" in domain:
          return self.handle_pubmed, resolved_url
      if "arxiv.org" in domain:
          return self.handle_arxiv, resolved_url
      if "biorxiv.org" in domain or "medrxiv.org" in domain:
          return self.handle_biorxiv, resolved_url
      if "nature.com" in domain:
          return self.handle_nature, resolved_url
      if "science.org" in domain or "sciencemag.org" in domain:
          return self.handle_science, resolved_url
      if "cell.com" in domain:
          return self.handle_cell, resolved_url
      if "frontiersin.org" in domain:
          return self.handle_frontiers, resolved_url
      else:
        return self.handle_generic, resolved_url

      return handler, url



    # ---------------- public API ----------------

    def download(self, urls):
        results = []

        for url in urls:
            print("\n===================================")
            print("🔍 Processing:", url)

            handler, resolved_url = self.get_handler(url)
            res = handler(resolved_url)
            results.append(res)

            if res.success:
                print("✅ Downloaded:", res.local_path)
            elif res.fallback_url:
                print("⚠️ Publisher page (manual download):")
                print("🔗", res.fallback_url)
            else:
                print("❌ Failed:", res.reason)

        return results
