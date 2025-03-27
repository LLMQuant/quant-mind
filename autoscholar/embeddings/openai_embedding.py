from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from autoscholar.embeddings import BaseEmbedding
from autoscholar.types import EmbeddingModelType


class OpenAIEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities using OpenAI's models.

    Attributes:
    ----------
        model_type (EmbeddingModelType, optional): The model type to be
            used for text embeddings.
            (default: :obj:`TEXT_EMBEDDING_3_SMALL`)
        url (Optional[str], optional): The url to the OpenAI service.
            (default: :obj:`None`)
        api_key (str, optional): The API key for authenticating with the
            OpenAI service. (default: :obj:`None`)
        dimensions (int, optional): The text embedding output dimensions.
            (default: :obj:`NOT_GIVEN`)

    Raises:
        RuntimeError: If an unsupported model type is specified.
    """

    def __init__(
        self,
        model_type: EmbeddingModelType = (
            EmbeddingModelType.TEXT_EMBEDDING_3_SMALL
        ),
        url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        if not model_type.is_openai:
            raise ValueError("Invalid OpenAI embedding model type.")
        self.model_type = model_type
        if dimensions is None:
            self.output_dim = model_type.output_dim
        else:
            assert isinstance(dimensions, int)
            self.output_dim = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._url = url or os.environ.get("OPENAI_API_BASE_URL")
        self.client = OpenAI(
            timeout=180,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )

    def embed_batch(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts.

        Args:
            objs (list[str]): The texts for which to generate the embeddings.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
            list[list[float]]: A list that represents the generated embedding
                as a list of floating-point numbers.
        """
        if self.model_type == EmbeddingModelType.TEXT_EMBEDDING_ADA_2:
            response = self.client.embeddings.create(
                input=objs,
                model=self.model_type.value,
                **kwargs,
            )
        else:
            response = self.client.embeddings.create(
                input=objs,
                model=self.model_type.value,
                dimensions=self.output_dim,
                **kwargs,
            )
        return [data.embedding for data in response.data]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim
