import logging
import os
from bs4 import BeautifulSoup
from task_oriented_dataset_search.preprocessing.interface import BaseLoader, Document
import fitz
import markdownify

logger = logging.getLogger(__name__)


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
        logger.debug(f"Loading text file using TextLoader: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Successfully loaded text file: {path}")
            return Document(text=text, metadata={"path": path})
        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
            raise e


class Pdf2MdLoader(BaseLoader):
    def load(self, path: str) -> Document:
        logger.debug(f"Loading PDF and converting to Markdown: {path}")
        try:
            doc = fitz.open(path)
            md_pages = []
            page_count = 0
            for page in doc:
                page_count += 1
                html_page = page.get_text("html")
                soup = BeautifulSoup(html_page, "html.parser")
                for img in soup.find_all("img"):
                    img.decompose()
                clean_html = str(soup)
                md_page = markdownify.markdownify(clean_html)
                md_pages.append(md_page)
            full_md = "\n\n".join(md_pages)
            logger.debug(
                f"Successfully converted PDF ({page_count} pages) to Markdown: {path}"
            )
            return Document(text=full_md, metadata={"path": path})
        except Exception as e:
            logger.error(f"Failed to convert PDF {path} to Markdown: {e}")
            raise


class PdfLoader(BaseLoader):
    def load(self, path: str) -> Document:
        logger.debug(f"Loading PDF text using PdfLoader: {path}")
        texts = []
        try:
            with fitz.open(path) as doc:
                page_count = 0
                for page in doc:
                    page_count += 1
                    page_text = page.get_text("text")
                    texts.append(page_text)
                full_text = "\n".join(texts)
            logger.debug(
                f"Successfully loaded text from PDF ({page_count} pages): {path}"
            )
            return Document(text=full_text, metadata={"path": path})
        except Exception as e:
            logger.error(f"Failed to load text from PDF {path}: {e}")
            raise


def get_loader(path: str) -> BaseLoader:
    ext = os.path.splitext(path)[1].lower()
    logger.debug(f"Getting loader for file: {path} (extension: {ext})")
    if ext == ".pdf":
        logger.debug("Selected PdfLoader.")
        return PdfLoader()
    elif ext in TextLoader.EXTS:
        logger.debug("Selected TextLoader.")
        return TextLoader()
    else:
        logger.error(f"Unsupported file extension: {ext} for path: {path}")
        raise ValueError(f"Unsupported file extension: {ext}")
