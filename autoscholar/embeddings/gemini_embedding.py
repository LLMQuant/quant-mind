# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from __future__ import annotations

import os
from typing import Any

from autoscholar.embeddings.base import BaseEmbedding
from autoscholar.types import EmbeddingModelType


class GeminiEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities using Google's Gemini models.

    https://ai.google.dev/gemini-api/docs/embeddings

    Attributes:
    ----------
        model_type (EmbeddingModelType, optional): The model type to be
            used for text embeddings.
            (default: :obj:`GEMINI_EMBEDDING`)
        api_key (str, optional): The API key for authenticating with the
            Google Gemini service. (default: :obj:`None`)
        dimensions (int, optional): The text embedding output dimensions.
            (default: :obj:`None`)
        task_type (str, optional): The type of task for which embeddings are optimized.
            Options include "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING",
            "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", etc. (default: :obj:`None`)

    Raises:
        ValueError: If an unsupported model type is specified.
    """

    def __init__(
        self,
        model_type: EmbeddingModelType = (
            EmbeddingModelType.GEMINI_EXP_EMBEDDING
        ),
        api_key: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        if not model_type.is_gemini:
            raise ValueError("Invalid Gemini embedding model type.")

        self.model_type = model_type
        self.task_type = task_type

        if dimensions is None:
            self.output_dim = model_type.output_dim
        else:
            assert isinstance(dimensions, int)
            self.output_dim = dimensions

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=self._api_key)
        self._config_type = types.EmbedContentConfig

    def embed_batch(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts in a batch.

        Parameters:
        ----------
            objs (list[str]): The texts for which to generate the embeddings.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
        -------
            list[list[float]]: A list that represents the
                generated embedding as a list of floating-point numbers.
        """
        config = None
        if self.task_type:
            config = self._config_type(task_type=self.task_type)

        response = self._client.models.embed_content(
            model=self.model_type.value,
            contents=objs,
            config=config,
            **kwargs,
        )

        embeddings = [embedding.values for embedding in response.embeddings]

        return embeddings

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
        -------
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim
