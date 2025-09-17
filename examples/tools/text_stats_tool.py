"""Text statistics tool demo."""

from quantmind.tools import tool


@tool
def summarize_text(text: str) -> dict:
    """Compute simple statistics for a text snippet.

    Args:
        text (str): Input paragraph to analyze.

    Returns:
        dict: Contains character and word counts.
    """
    words = [word for word in text.split() if word]
    return {
        "characters": len(text),
        "words": len(words),
    }


def main():
    """Run the text statistics tool with a sample snippet."""
    sample = "Markets rally as investors digest the latest earnings reports."
    stats = summarize_text(sample)
    print(f"Text: {sample}")
    print(f"Characters: {stats['characters']}, Words: {stats['words']}")


if __name__ == "__main__":
    main()
