# Knowledge Graph from Academic Paper Abstracts

This demo shows how to build and visualize a knowledge graph based on academic paper abstracts using AutoScholar.

![demo-kg-abstract](../../asset/demo-kg-abstract.png)

## Overview

This demonstration:

1. Loads a collection of academic papers from JSON files
2. Uses pre-computed embeddings to represent each paper
3. Builds a knowledge graph connecting related papers
4. Visualizes the graph as an interactive HTML page (similar to ConnectedPapers)

## How It Works

The demo follows these steps:

1. Loads JSON files containing paper metadata
2. Retrieves the corresponding pre-computed embeddings for each paper
3. Uses `KnowledgeGraphBuilder` to construct a graph that connects related papers
4. Visualizes the resulting graph using `GraphVisualizer` with Pyvis

The visualization allows you to explore paper relationships interactively, with physics settings customized to create an intuitive layout.

## Usage

```bash
cd our-repo-root
python examples/kg_by_abstract/demo.py
```

This will generate an interactive HTML file named `paper_network.html`.

## Limitations

- The quality of paper connections depends entirely on the **quality of the embeddings**
- The demo does not include methods for filtering or categorizing papers by topic
- The visualization may become cluttered with large numbers of papers

The demo is intended as a starting point for building more sophisticated knowledge graph applications with academic papers.
