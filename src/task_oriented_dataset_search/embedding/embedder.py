import logging
import time
from typing import List

import numpy as np
from openai import OpenAI
from task_oriented_dataset_search.embedding.interface import BaseEmbedder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        logger.info(
            f"Initializing SentenceTransformerEmbedder with model: {model_name}"
        )
        logger.debug(f"Device: {device or 'auto'}, Batch size: {batch_size}")
        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self.batch_size = batch_size
            logger.info(
                f"SentenceTransformer model loaded successfully. Dimension: {self._dimension}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model {model_name}: {e}",
                exc_info=True,
            )
            raise

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("Embed called with an empty list of texts.")
            return np.array([], dtype="float32").reshape(0, self._dimension)

        logger.debug(f"Embedding {len(texts)} texts...")
        start_time = time.time()

        try:
            embs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            end_time = time.time()
            logger.info(
                f"Finished embedding {len(texts)} texts in {end_time - start_time:.2f} seconds."
            )
            return embs.astype("float32")
        except Exception as e:
            logger.error(f"Failed during text embedding: {e}", exc_info=True)
            raise


class OpenAIAPITextEmbedder(BaseEmbedder):
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        api_base: str | None = None,
        batch_size: int = 100,
    ):
        logger.info(f"Initializing OpenAIAPITextEmbedder with model: {model_name}")
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model_name = model_name
        if model_name == "text-embedding-3-small":
            self._dimension = 1536
        elif model_name == "text-embedding-3-large":
            self._dimension = 3072
        elif model_name == "text-embedding-ada-002":
            self._dimension = 1536
        else:
            logger.warning(
                f"Dimension for model '{model_name}' is not pre-defined. "
                f"Defaulting to 1536 (as for 'text-embedding-3-small'). "
                f"Ensure this is correct for your model."
            )
            self._dimension = 1536
        self.batch_size = batch_size
        logger.info(
            f"OpenAIAPITextEmbedder model set to: {self._model_name}. Dimension: {self._dimension}"
        )

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("Embed called with an empty list of texts.")
            return np.array([], dtype="float32").reshape(0, self._dimension)

        all_embeddings: List[List[float]] = []
        num_texts = len(texts)

        logger.debug(
            f"Embedding {num_texts} texts using OpenAI model '{self._model_name}'..."
        )
        start_time_total = time.time()

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            processed_batch_texts = [text.replace("\n", " ") for text in batch_texts]
            logger.debug(
                f"Processing batch {i//self.batch_size + 1}/{(num_texts - 1)//self.batch_size + 1} with {len(processed_batch_texts)} texts."
            )
            start_time_batch = time.time()

            try:
                response = self._client.embeddings.create(
                    input=processed_batch_texts,
                    model=self._model_name,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                end_time_batch = time.time()
                logger.debug(
                    f"Batch {i//self.batch_size + 1} embedded in {end_time_batch - start_time_batch:.2f} seconds."
                )

            except Exception as e:
                logger.error(
                    f"Failed during OpenAI API text embedding for a batch: {e}",
                    exc_info=True,
                )
                raise

        end_time_total = time.time()
        logger.info(
            f"Finished embedding {num_texts} texts in {end_time_total - start_time_total:.2f} seconds using {self._model_name}."
        )

        return np.array(all_embeddings, dtype="float32")
