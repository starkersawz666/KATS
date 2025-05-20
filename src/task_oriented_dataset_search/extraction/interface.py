from typing import Dict, Any

ExtractionResult = Dict[str, Any]


class BaseExtractor:
    def extract(self, text: str, **kwargs) -> ExtractionResult:
        raise NotImplementedError
