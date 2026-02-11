import os
import re
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from pypdf import PdfReader


class PaperSummarizer:

    def __init__(self, model_name="google/pegasus-pubmed"):

        self.MODEL_NAME = model_name

        self.tokenizer = PegasusTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)


    # ---------- PDF LOADER ----------
    def load_paper(self, file_path):

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text


    # ---------- SECTION SPLITTER ----------
    def split_sections(self, text):

        sections = {
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": ""
        }

        patterns = {
            "introduction": r"(introduction)(.*?)(materials|methods|results|discussion)",
            "methods": r"(materials and methods|methods)(.*?)(results|discussion)",
            "results": r"(results)(.*?)(discussion)",
            "discussion": r"(discussion)(.*)"
        }

        text = text.lower()

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[key] = match.group(2)

        return sections


    # ---------- SUMMARIZER ----------
    def summarize_text(self, text, max_length=256):

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        summary_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    # ---------- SUMMARIZE ONE PAPER ----------
    def summarize_paper(self, file_path):

        if not os.path.exists(file_path):
            print("❌ File not found")
            return

        print("✅ Loading paper...")
        text = self.load_paper(file_path)

        print("✅ Splitting sections...")
        sections = self.split_sections(text)

        print("""
Choose:
1 - Introduction
2 - Methods
3 - Results
4 - Discussion
5 - Full paper summary
""")

        choice = input("Your choice: ")

        summaries = {}

        for sec in sections:
            if sections[sec].strip():
                print(f"Summarizing {sec}...")
                summaries[sec] = self.summarize_text(sections[sec])

        if choice == "1":
            return summaries["introduction"]

        elif choice == "2":
            return summaries["methods"]

        elif choice == "3":
            return summaries["results"]

        elif choice == "4":
            return summaries["discussion"]

        elif choice == "5":
            combined = " ".join(summaries.values())
            return self.summarize_text(combined, 350)

        else:
            return "Invalid choice"


    # ---------- SELECT FROM DOWNLOADED PAPERS ----------
    def summarize_from_folder(self, folder_path):

        files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        if not files:
            print("❌ No PDFs found")
            return

        print("\nAvailable papers:")
        for i, f in enumerate(files):
            print(f"{i+1}. {f}")

        idx = int(input("\nSelect paper number: ")) - 1

        file_path = os.path.join(folder_path, files[idx])

        summary = self.summarize_paper(file_path)

        print("\n✅ SUMMARY:\n")
        print(summary)
