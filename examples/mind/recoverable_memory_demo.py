"""Minimal demo for recoverable ingestion and gated memory commit."""

from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict

from quantmind.configs.recoverable import RecoverableValidation
from quantmind.flows.governance import load_governance_policy
from quantmind.mind.memory import HybridMemoryEngine


class DemoArtifact(BaseModel):
    """Strict schema that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    validation_confidence: float
    provenance: dict[str, str]


def main() -> None:
    validator = RecoverableValidation(DemoArtifact)
    policy = load_governance_policy()
    payload = {
        "id": "artifact-1",
        "title": "Recovered architecture summary",
        "validation_confidence": 0.91,
        "provenance": {"source": "demo"},
        "unexpected_field": "will trigger fallback",
    }

    with TemporaryDirectory() as tmpdir:
        memory = HybridMemoryEngine(base_path=tmpdir)
        parsed = validator.execute_safely(payload)
        memory.write_l1_trace("Parser", parsed.model_dump(mode="json"))

        if isinstance(parsed, DemoArtifact):
            committed = memory.commit_to_l3_graph(
                parsed.model_dump(mode="json"),
                schema_valid=True,
                dedup_ok=True,
                contradiction_free=True,
            )
            print("Committed to L3:", committed)
        else:
            fallback_node = parsed
            fallback = policy.global_settings.fallback_policy
            print("Fallback policy:", fallback)
            print("Quarantined schema:", fallback_node.target_schema)


if __name__ == "__main__":
    main()
