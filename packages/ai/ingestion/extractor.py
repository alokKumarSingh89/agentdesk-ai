from pathlib import Path

from pypdf import PdfReader

class DocumentTextExtractor:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
    
    def extract(self, file_path: Path) -> str:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )
        if extension == ".txt":
            text = file_path.read_text(
                encoding="utf-8"
            )
        else:
            reader = PdfReader(str(file_path))
            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]
            
            text = "\n\n".join(pages)
        text = text.strip()
        
        if not text:
            raise ValueError(
                "Document contains no extractable text."
            )
        return text
            