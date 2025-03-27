from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseEmbedding(ABC, Generic[T]):
    r"""Abstract base class for text embedding functionalities."""

    @abstractmethod
    def embed_batch(
        self,
        objs: list[T],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts in a batch.

        Parameters:
        ----------
            objs (list[T]): The objects for which to generate the embeddings.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
        -------
            list[list[float]]: A list that represents the
                generated embedding as a list of floating-point numbers.
        """
        pass

    def embed(
        self,
        obj: T,
        **kwargs: Any,
    ) -> list[float]:
        r"""Generates an embedding for the given text.

        Parameters:
        ----------
            obj (T): The object for which to generate the embedding.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
        -------
            list[float]: A list of floating-point numbers representing the
                generated embedding.
        """
        return self.embed_batch([obj], **kwargs)[0]

    @abstractmethod
    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
        -------
            int: The dimensionality of the embedding for the current model.
        """
        pass
