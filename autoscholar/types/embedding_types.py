from enum import Enum


class EmbeddingModelType(Enum):
    """Embedding model types."""

    TEXT_EMBEDDING_ADA_2 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_EXP_EMBEDDING = "gemini-embedding-exp-03-07"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            EmbeddingModelType.TEXT_EMBEDDING_ADA_2,
            EmbeddingModelType.TEXT_EMBEDDING_3_SMALL,
            EmbeddingModelType.TEXT_EMBEDDING_3_LARGE,
        }

    @property
    def is_gemini(self) -> bool:
        if self in {
            EmbeddingModelType.GEMINI_EXP_EMBEDDING,
        }:
            return True
        else:
            return False

    @property
    def output_dim(self) -> int:
        if self in {
            EmbeddingModelType.GEMINI_EXP_EMBEDDING,
        }:
            return 3072
        elif self is EmbeddingModelType.TEXT_EMBEDDING_ADA_2:
            return 1536
        elif self is EmbeddingModelType.TEXT_EMBEDDING_3_SMALL:
            return 1536
        elif self is EmbeddingModelType.TEXT_EMBEDDING_3_LARGE:
            return 3072
        else:
            raise ValueError(f"Unknown model type {self}.")
