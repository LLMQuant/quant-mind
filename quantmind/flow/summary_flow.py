"""Content summary generation flow using two-stage approach for QuantMind framework."""

from typing import List

from quantmind.config.flows import ChunkingStrategy
from quantmind.flow.base import BaseFlow
from quantmind.models.content import KnowledgeItem
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryFlow(BaseFlow):
    """A two-step built-in summary flow: chunk documents, then combine summaries.

    You can set `use_chunking` to False to use the powerful model directly on the full content or
    change the chunking strategy to `ChunkingStrategy.BY_CUSTOM` to use a custom chunking strategy.

    This flow demonstrates the new architecture by implementing a cost-effective
    approach: use a cheap model for chunk summarization, then a powerful model
    for final combination.
    """

    def run(self, document: KnowledgeItem) -> str:
        """Execute the two-stage summary process.

        Args:
            document: KnowledgeItem to summarize

        Returns:
            Final combined summary
        """
        logger.info(f"Starting summary flow for: {document.title}")

        content = document.content or ""
        if not content:
            logger.warning("No content to summarize")
            return "No content available for summarization."

        # Two different strategies based on chunking configuration
        if self.config.use_chunking:
            # Strategy 1: Chunking mode - use cheap model for chunks, powerful for combination
            logger.debug("Using chunking mode with two-stage summarization")

            chunks = self._chunk_document(content)
            if not chunks:
                logger.warning("No chunks generated")
                return "No content available for summarization."

            # Use cheap model to summarize each chunk
            summarizer_llm = self._llm_blocks["cheap_summarizer"]
            chunk_summaries = []

            for i, chunk in enumerate(chunks):
                logger.debug(f"Summarizing chunk {i + 1}/{len(chunks)}")
                prompt = self._render_prompt(
                    "summarize_chunk_template", chunk_text=chunk
                )
                summary = summarizer_llm.generate_text(prompt)
                if summary:
                    chunk_summaries.append(summary)

            if not chunk_summaries:
                logger.error("Failed to generate any chunk summaries")
                return "Failed to summarize content."

            # If only one chunk, return its summary directly
            if len(chunk_summaries) == 1:
                logger.info(
                    f"Successfully generated summary for: {document.title}"
                )
                return chunk_summaries[0]

            # Use powerful model to combine multiple chunk summaries
            combiner_llm = self._llm_blocks["powerful_combiner"]
            final_prompt = self._render_prompt(
                "combine_summaries_template",
                summaries="\n\n".join(chunk_summaries),
            )

            final_summary = combiner_llm.generate_text(final_prompt)

            if final_summary:
                logger.info(
                    f"Successfully generated summary for: {document.title}"
                )
                return final_summary
            else:
                logger.error("Failed to generate final summary")
                return "Failed to generate final summary."

        else:
            # Strategy 2: No chunking - use powerful model directly on full content
            logger.debug(
                "Using non-chunking mode with direct powerful model summarization"
            )

            combiner_llm = self._llm_blocks["powerful_combiner"]
            prompt = self._render_prompt(
                "summarize_chunk_template", chunk_text=content
            )

            summary = combiner_llm.generate_text(prompt)

            if summary:
                logger.info(
                    f"Successfully generated summary for: {document.title}"
                )
                return summary
            else:
                logger.error("Failed to generate summary")
                return "Failed to summarize content."

    def _chunk_document(self, text: str) -> List[str]:
        """Split document into chunks for processing.

        Args:
            text: Document text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []
        if self.config.chunk_strategy == ChunkingStrategy.BY_CUSTOM:
            if self.config.chunk_custom_strategy:
                return self.config.chunk_custom_strategy(text)
            else:
                logger.warning(
                    "Custom chunking strategy specified but no function provided, falling back to BY_SIZE"
                )
        elif self.config.chunk_strategy == ChunkingStrategy.BY_SECTION:
            raise NotImplementedError(
                "Chunking by section is not implemented for this flow."
            )

        # Default to BY_SIZE strategy (already validated in config)
        chunk_size = self.config.chunk_size
        chunks = []

        # Simple chunking by character count with word boundary preservation
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]

            # Try to end at word boundary if not last chunk
            if i + chunk_size < len(text):
                last_space = chunk.rfind(" ")
                if (
                    last_space > chunk_size // 2
                ):  # Only if we don't lose too much
                    chunk = chunk[:last_space]

            chunks.append(chunk.strip())

        logger.debug(
            f"Split document into {len(chunks)} chunks using {self.config.chunk_strategy.value} strategy"
        )
        return chunks
