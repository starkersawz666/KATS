from typing import Dict

class Document:
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata
    
class BaseLoader:
    def load(self, path: str) -> Document:
        raise NotImplementedError