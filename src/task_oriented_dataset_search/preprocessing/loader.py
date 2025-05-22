import os
from bs4 import BeautifulSoup
from task_oriented_dataset_search.preprocessing.interface import BaseLoader, Document
import fitz
import markdownify


class TextLoader(BaseLoader):
    EXTS = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".rst",
        ".tex",
        ".log",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
    }

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return Document(text=text, metadata={"path": path})


class Pdf2MdLoader(BaseLoader):
    def load(self, path: str) -> Document:
        doc = fitz.open(path)
        md_pages = []
        for page in doc:
            html_page = page.get_text("html")
            soup = BeautifulSoup(html_page, "html.parser")
            for img in soup.find_all("img"):
                img.decompose()
            clean_html = str(soup)
            md_page = markdownify.markdownify(clean_html)
            md_pages.append(md_page)
        full_md = "\n\n".join(md_pages)
        return Document(text=full_md, metadata={"path": path})


class PdfLoader(BaseLoader):
    def load(self, path: str) -> Document:
        texts = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                texts.append(page_text)
        full_text = "\n".join(texts)
        return Document(text=full_text, metadata={"path": path})


def get_loader(path: str) -> BaseLoader:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PdfLoader()
    elif ext in TextLoader.EXTS:
        return TextLoader()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
