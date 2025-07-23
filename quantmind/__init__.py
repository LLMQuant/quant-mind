"""QuantMind: Intelligent Knowledge Extraction and Retrieval Framework.

QuantMind transforms unstructured financial content into a queryable knowledge graph
through a two-stage architecture focused on knowledge extraction and intelligent retrieval.
"""

__version__ = "0.2.0"

# Import core models
from quantmind.models.paper import Paper
from quantmind.models.knowledge_graph import KnowledgeGraph

__all__ = ["Paper", "KnowledgeGraph"]

# Conditionally import workflow components
try:
    from quantmind.workflow.agent import WorkflowAgent

    __all__.append("WorkflowAgent")
except ImportError:
    pass
