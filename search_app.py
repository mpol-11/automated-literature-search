import os
import pypdf
import re
import requests
import pandas as pd
import urllib.parse
import time
import feedparser
import xml.etree.ElementTree as ET
import torch
import sentencepiece

# Mount Drive (once per runtime)
# ===================== OPTIONAL COLAB SETUP =====================
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True

    # Mount Drive (only in Colab)
    drive.mount('/content/drive')

    import sys
    sys.path.append('/content/drive/MyDrive/colab_modules')

except ImportError:
    # Not running in Colab (GitHub / local)
    pass


from downloader import PaperDownloader, DownloadResult
from summarizer import PaperSummarizer


# Correcting the import path for PaperDownloader from the cloned repository
'''import sys
sys.path.insert(0, '/content/pubmed_pdf_downloader') # Add the root of the cloned repo to sys.path
from paper_downloader import PaperDownloader'''



from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pathlib import Path
from unittest.mock import patch
from IPython.core.debugger import set_trace
try:
    from google.colab import auth
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except ImportError:
    pass

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM




# ===================== CONSTANTS =====================
FETCH_SIZE = 300       # how many papers to retrieve per iteration
DISPLAY_SIZE = 30      # how many papers to show to the user

# ===================== EMBEDDING CACHE =====================
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
EMBEDDING_CACHE = {}  # paper_id -> embedding

SEARCH_TOPIC = "topic"
SEARCH_AUTHOR = "author"
SEARCH_BOTH = "both"


# ===================== FOLDER CREATION =====================
def create_folder(folder_name, drive_folder):
    parent_folder = "/content/drive/MyDrive"
    drive_folder = os.path.join(parent_folder, folder_name)
    os.makedirs(drive_folder, exist_ok=True)
    return drive_folder


def create_folder_desktop(folder_name, computer_folder):
    parent_folder = "path_to_folder"
    desktop_folder = os.path.join(parent_folder, folder_name)
    os.makedirs(desktop_folder, exist_ok=True)
    return desktop_folder



# ===================== YEAR FILTER =====================
def build_year_filter():
    choice = input("Do you want to filter by publication year(s)? (yes/no): ").strip().lower()
    if choice != "yes":
        return ""

    start_year = input("Enter start year (YYYY): ").strip()
    end_year = input("Enter end year (YYYY) or press Enter for a single year: ").strip()

    if end_year:
        return f" AND ({start_year}:{end_year}[pdat])"
    return f" AND ({start_year}[pdat])"

# ===================== TOPIC FILTER =====================
def parse_topics(topic_input):
    topics = [t.strip().lower() for t in topic_input.split(",") if t.strip()]
    return topics

def contains_all_keywords(text, keywords):
    text = text.lower()
    return all(k in text for k in keywords)

# ===================== PUBMED SEARCH (PAGINATED) =====================
def build_pubmed_query(topic_input):
    topics = parse_topics(topic_input)

    if len(topics) == 1:
        return topics[0]

    return " AND ".join([f"\"{t}\"[Title/Abstract]" for t in topics])

def search_pubmed(topic, year_filter="", max_results=100, retstart=0):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    topic_query = build_pubmed_query(topic)
    query = f"{topic_query}{year_filter}"


    query_url = (
        f"{base_url}?db=pubmed"
        f"&term={urllib.parse.quote_plus(query)}"
        f"&retstart={retstart}"
        f"&retmax={max_results}"
        f"&retmode=json&sort=relevance"
    )

    response = requests.get(query_url)
    if response.status_code != 200:
        return []

    return response.json()["esearchresult"]["idlist"]


def search_papers_by_author(author, year_filter="", max_results=FETCH_SIZE, retstart=0):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    query = f"{author}{year_filter}"

    query_url = (
        f"{base_url}?db=pubmed"
        f"&term={urllib.parse.quote_plus(query)}"
        f"&retstart={retstart}"
        f"&retmax={max_results}"
        f"&retmode=json&sort=relevance"
    )

    response = requests.get(query_url)
    if response.status_code == 200:
        return response.json()["esearchresult"]["idlist"]
    return []


def search_papers_by_topic_and_author(topic, author, year_filter="", max_results=FETCH_SIZE, retstart=0):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    query = f"\"{topic}\"[Title/Abstract] AND \"{author}\"[Author]{year_filter}"

    query_url = (
        f"{base_url}?db=pubmed"
        f"&term={urllib.parse.quote_plus(query)}"
        f"&retstart={retstart}"
        f"&retmax={max_results}"
        f"&retmode=json&sort=relevance"
    )

    response = requests.get(query_url)
    if response.status_code != 200:
        return []

    return response.json()["esearchresult"]["idlist"]


def fetch_paper_metadata(pmids):
    if not pmids:
        return None

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(pmids)}&retmode=xml"
    )
    return requests.get(url).text


def extract_abstracts_pubmed(xml_data):
    papers = {}
    if not xml_data:
        return papers

    root = ET.fromstring(xml_data)

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = ''.join(article.find(".//ArticleTitle").itertext())
        abstract = article.findtext(".//Abstract/AbstractText", "Abstract not available")

        authors = []
        for author in article.findall(".//Author"):
            authors.append(f"{author.findtext('ForeName','')} {author.findtext('LastName','')}".strip())

        papers[pmid] = {
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "source": "pubmed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }

    return papers


# ===================== ARXIV & BIORXIV =====================

def search_arxiv(topic=None, author=None, search_mode=SEARCH_TOPIC, max_results=FETCH_SIZE):
    # 1. Build a BROAD query (never rely on arXiv for strict logic)
    if search_mode == SEARCH_AUTHOR:
        query = f"au:{author}"
    else:
        query = f"all:{topic}"

    if search_mode == SEARCH_BOTH:
        query = f"all:{topic} AND au:{author}"

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote_plus(query)}"
        f"&max_results={max_results * 5}"
    )

    feed = feedparser.parse(url)
    papers = {}

    for entry in feed.entries:
        title = entry.title.lower()
        abstract = entry.summary.lower()
        authors = [a.name.lower() for a in entry.authors]

        # 2. Enforce STRICT logic in Python
        topics = parse_topics(topic)

        if search_mode in (SEARCH_TOPIC, SEARCH_BOTH):
          if not (
              contains_all_keywords(title, topics)
              or contains_all_keywords(abstract, topics)
          ):
            continue


        if search_mode in (SEARCH_AUTHOR, SEARCH_BOTH):
            if not any(author.lower() in a for a in authors):
                continue

        arxiv_id = entry.id.split("/")[-1]
        papers[arxiv_id] = {
            "id": arxiv_id,
            "title": entry.title,
            "abstract": entry.summary,
            "authors": ", ".join(a.name for a in entry.authors),
            "source": "arxiv",
            "url": entry.link
        }

        if len(papers) >= max_results:
            break

    return papers



def search_biorxiv(topic=None, author=None, search_mode=SEARCH_TOPIC, max_results=FETCH_SIZE):
    url = "https://api.biorxiv.org/details/biorxiv/2000-01-01/2100-01-01"
    response = requests.get(url)
    papers = {}

    if response.status_code != 200:
        return papers

    for item in response.json()["collection"]:
        title = item["title"].lower()
        abstract = item["abstract"].lower()
        authors = item["authors"].lower()

        # STRICT logic
        topics = parse_topics(topic)

        if search_mode in (SEARCH_TOPIC, SEARCH_BOTH):
            if not (
              contains_all_keywords(title, topics)
              or contains_all_keywords(abstract, topics)
            ):
              continue


        if search_mode in (SEARCH_AUTHOR, SEARCH_BOTH):
            if author.lower() not in authors:
                continue

        doi = item["doi"]
        papers[doi] = {
            "id": doi,
            "title": item["title"],
            "abstract": item["abstract"],
            "authors": item["authors"],
            "source": "biorxiv",
            "url": f"https://www.biorxiv.org/content/{doi}.full"
        }

        if len(papers) >= max_results:
            break

    return papers

# ===================== SEMANTIC FILTER =====================
def improve_search_relevance_semantic(topic, papers_data, top_n=DISPLAY_SIZE):
    if not papers_data:
        return {}

    topics = parse_topics(topic)
    topic_embedding = EMBEDDING_MODEL.encode(" ".join(topics))

    scores = {}

    for pid, paper in papers_data.items():
        abstract = paper.get("abstract")
        if not abstract or abstract == "Abstract not available":
            continue

        if pid not in EMBEDDING_CACHE:
            EMBEDDING_CACHE[pid] = EMBEDDING_MODEL.encode(abstract)

        scores[pid] = cosine_similarity(
            [topic_embedding],
            [EMBEDDING_CACHE[pid]]
        )[0][0]

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return {pid: papers_data[pid] for pid in sorted_ids}

def summarize_abstract(abstract_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    if abstract_text and abstract_text.strip():
        try:
            summary = summarizer(
                abstract_text,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error summarizing results: {e}")
            return "Error summarizing results."
    else:
        return "No results to summarize."

# PDF LINK
def find_pdf_link(paper):
    if paper["source"] == "pubmed":
        base_url = paper["url"]
        response = requests.get(base_url)
        if response.status_code == 200 and "doi.org" in response.text:
            start = response.text.find("https://doi.org/")
            end = response.text.find('"', start)
            return response.text[start:end]
        return base_url
    else:
        return paper["url"]

def generate_report(papers, topic):
    report = f"Scientific Topic Review: {topic}\n\n"

    for paper in papers.values():
        report += f"## {paper['title']}\n"
        report += f"### Summary of Abstract:\n{paper.get('summary', 'Not available')}\n"
        report += f"Authors: {paper.get('authors', 'Not available')}\n"
        report += f"Source: {paper.get('source')}\n"
        report += f"Link: {paper.get('url', 'No URL available')}\n\n"

    return report

#USER SELECTION

def check_criteria(papers_data):
    selected_ids = []

    for paper_id, paper in papers_data.items():
        decision = input(
            f"Are you satisfied with the paper '{paper['title']}' (ID: {paper_id})? (yes/no): "
        )
        if decision.lower() == 'yes':
            selected_ids.append(paper_id)
            print(f"✅ Added paper {paper_id} to the selection list.")

    return selected_ids

# ===================== UTILS =====================
def filter_new_papers(paper_ids, shown_ids):
    return [pid for pid in paper_ids if pid not in shown_ids]


def ask_user_satisfaction():
    while True:
        answer = input("\nAre you satisfied with the current search results? (yes/no): ").strip().lower()
        if answer in ("yes", "no"):
            return answer


# ===================== MAIN =====================
def main(topic, max_results=DISPLAY_SIZE):
    shown_ids = set()
    selected_ids = set()
    pubmed_offset = 0

    chosen_papers = []


    folder_to_create = input("Where do you want to create the folder: (select 1 or 2): ")
    folder_name = "Downloaded_papers"

    if folder_to_create == "1":
        drive_folder = "/content/drive/MyDrive"
        output_folder= create_folder(folder_name, drive_folder)
    elif folder_to_create == "2":
        computer_folder = "path_to_folder"
        output_folder= create_folder_desktop(folder_name, computer_folder)

    author = input("Enter the author name (or leave empty): ").strip()

    search_parameters = input("Which search parameter do you want to use: 1=topic, 2=author, 3=both: ")
    search_source = input("Which source do you want to search in: 1=PubMed, 2=arXiv, 3=bioRxiv, 4=ALL: ")
    year_filter = build_year_filter()

    if search_parameters == "1":
        search_mode = "topic"
    elif search_parameters == "2":
        search_mode = "author"
    else:
        search_mode = "both"

    satisfied = False

    while not satisfied:
        papers_data = {}

        if search_source in ("1", "4"):

            if search_mode == SEARCH_TOPIC:
                pmids = search_pubmed(
                    topic,
                    year_filter,
                    max_results=FETCH_SIZE,
                    retstart=pubmed_offset
                )

            elif search_mode == SEARCH_AUTHOR:
                pmids = search_papers_by_author(
                    author,
                    year_filter,
                    max_results=FETCH_SIZE,
                    retstart=pubmed_offset
                )

            else:  # SEARCH_BOTH
                pmids = search_papers_by_topic_and_author(
                    topic,
                    author,
                    year_filter,
                    max_results=FETCH_SIZE,
                    retstart=pubmed_offset
                )

            pubmed_offset += FETCH_SIZE
            pmids = filter_new_papers(pmids, shown_ids)

            xml_data = fetch_paper_metadata(pmids)
            papers_data.update(extract_abstracts_pubmed(xml_data))


        if search_source in ("2", "4"):
            papers_data.update(search_arxiv(topic, author, search_mode))

        if search_source in ("3", "4"):
            papers_data.update(search_biorxiv(topic, author, search_mode))

        if not papers_data:
            print("🚫 No papers found.")

            while True:
              retry = input("\nDo you want to start a new search? (yes/no): ").strip().lower()

              if retry == "yes":
                print("\n🔄 Restarting search...\n")

                new_topic = input("Enter the scientific topic for review: ")
                main(new_topic, max_results)   # restart whole program
                return   # stop current execution

              elif retry == "no":
                print("\n👋 Exiting application.")
                return

              else:
                print("❌ Please type 'yes' or 'no'.")



        if search_mode in ("topic", "both"):
            papers_data = improve_search_relevance_semantic(topic, papers_data)

        shown_ids.update(papers_data.keys())

        print(f"\n--- Reviewing {len(papers_data)} abstracts ---")

        selected_papers = []

        for idx, (paper_id, data) in enumerate(papers_data.items()):
            if idx >= 30:
                break

            print(f"\n**Paper ID: {paper_id} ({data['source']})**")
            print(f"Title: {data['title']}")
            print(f"Abstract: {data['abstract'][:500]}...")
            print(f"Authors: {data['authors']}")

            summary = summarize_abstract(data["abstract"])
            print(f"\nSummarized abstract:\n{summary}")
            print("=" * 80)

            pdf_link = find_pdf_link(data)

            selected_papers.append({
                "id": paper_id,
                "title": data["title"],
                "summary": summary,
                "url": pdf_link,
                "authors": data["authors"],
                "source": data["source"]
            })

        # ===================== USER SELECTION =====================
        chosen_ids = check_criteria(papers_data)
        chosen_papers = [p for p in selected_papers if p["id"] in chosen_ids]

        if chosen_papers:
            report = generate_report(
                {p["id"]: p for p in chosen_papers}, topic
            )
            print("\n--- Abstract Report ---\n")
            print(report)

        satisfied = ask_user_satisfaction() == "yes"

    print("\n✅ Search session completed.")

          #for pid, paper in list(papers_data.items())[:DISPLAY_SIZE]:
            #print(f"\n{paper['title']}\n{paper['abstract'][:500]}...\n")

    # ===================== DOWNLOAD SELECTED PAPERS =====================
    if chosen_papers and output_folder:
            download_choice = input(
                "\nDo you want to download the selected papers now? (yes/no): "
            ).strip().lower()

            if download_choice == "yes":
                print("\n📥 Attempting to download selected papers...\n")

                downloader = PaperDownloader(out_dir=output_folder)
                paper_urls = [p["url"] for p in chosen_papers]

                results = downloader.download(paper_urls)

                failed = [r for r in results if not r.success]
                if failed:
                    print("\n⚠️ Some papers require manual download:")
                    for r in failed:
                        if r.fallback_url:
                            print("🔗", r.fallback_url)



        # ===================== FULL PAPER SUMMARIZATION =====================

    while True:
      summarize_choice = input(
        "\nDo you want to summarize any downloaded paper? (yes/no): "
      ).strip().lower()

      if summarize_choice == "yes":
        summarizer = PaperSummarizer()

        print("\n📂 Opening downloaded papers folder...")
        summarizer.summarize_from_folder(output_folder)

      elif summarize_choice == "no":
        print("\n👋 Exiting summarization loop.")
        break

      else:
        print("❌ Invalid input. Please type 'yes' or 'no'.")


#TODO Add citation extraction feature



if __name__ == "__main__":
  topic= input("Enter the scientific topic for review: ")

  main(topic, max_results=30)