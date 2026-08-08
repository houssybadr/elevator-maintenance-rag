import torch
from FlagEmbedding import BGEM3FlagModel
from app.core.config import settings


class MockModel:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def encode(
        self,
        text,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ):
        _ = (max_length, return_dense, return_sparse, return_colbert_vecs)
        if isinstance(text, str):
            return {"dense_vecs": [0.0] * self.embedding_dim}
        return {"dense_vecs": [[0.0] * self.embedding_dim for _ in text]}


class Embedder:
    def __init__(self):
        self.model=None

    def load(self):
        if settings.skip_embedder_load:
            self.model = MockModel(settings.mock_embedding_dim)
            print("Mock embedding model loaded")
            return

        print("Loading the model")
        self.model=BGEM3FlagModel(
            settings.embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_fp16=True
        )
        print("Model Loaded")

    def embed(self,text:str) -> list:
        if self.model is None:
            self.load()

        result=self.model.encode(
            text,
            max_length=settings.max_tokens,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        dense_vecs = result["dense_vecs"]
        if hasattr(dense_vecs, "tolist"):
            return dense_vecs.tolist()
        return dense_vecs
    
