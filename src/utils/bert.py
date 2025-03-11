from sentence_transformers import SentenceTransformer


class BertService:

    def __init__(self):
        pass

    @staticmethod
    def text_to_vector(model: SentenceTransformer, text: str):
        return model.encode(text).astype("float32")
