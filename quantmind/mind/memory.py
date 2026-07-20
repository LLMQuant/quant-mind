"""Hybrid memory primitives for L1/L2/L3 data handling."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class L3CommitRequirements:
    """Hard gates required before durable L3 commit."""

    min_confidence: float = 0.85
    require_provenance: bool = True
    require_schema_validity: bool = True
    require_dedup_check: bool = True
    require_contradiction_check: bool = True


def can_commit_to_l3(
    artifact: dict[str, Any],
    *,
    schema_valid: bool,
    dedup_ok: bool,
    contradiction_free: bool,
    requirements: L3CommitRequirements | None = None,
) -> bool:
    """Return true only when artifact passes all required L3 gates."""
    req = requirements or L3CommitRequirements()
    confidence = float(artifact.get("validation_confidence", 0.0))
    if confidence < req.min_confidence:
        return False
    if req.require_provenance and not artifact.get("provenance"):
        return False
    if req.require_schema_validity and not schema_valid:
        return False
    if req.require_dedup_check and not dedup_ok:
        return False
    if req.require_contradiction_check and not contradiction_free:
        return False
    return True


class HybridMemoryEngine:
    """L1 (transient) -> L2 (working) -> L3 (durable) memory manager."""

    def __init__(self, base_path: str = "~/.quantmind/mind") -> None:
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.l1_path = self.base_path / "l1_ephemeral.jsonl"
        self.l2_path = self.base_path / "l2_working.json"
        self.l3_path = self.base_path / "l3_durable.jsonl"

    def write_l1_trace(self, agent_role: str, tool_output: Any) -> None:
        """Append raw tool output to L1 trace."""
        record = {"role": agent_role, "payload": tool_output}
        with self.l1_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def update_l2_state(self, key: str, value: Any) -> dict[str, Any]:
        """Update and persist L2 working state as JSON."""
        current: dict[str, Any] = {}
        if self.l2_path.exists():
            with self.l2_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
                if isinstance(loaded, dict):
                    current = loaded
        current[key] = value
        with self.l2_path.open("w", encoding="utf-8") as file:
            json.dump(current, file, ensure_ascii=False, indent=2, sort_keys=True)
        return current

    def commit_to_l3_graph(
        self,
        validated_artifact: dict[str, Any],
        *,
        schema_valid: bool,
        dedup_ok: bool,
        contradiction_free: bool,
        requirements: L3CommitRequirements | None = None,
    ) -> bool:
        """Commit artifact to L3 only when every hard gate passes."""
        if not can_commit_to_l3(
            validated_artifact,
            schema_valid=schema_valid,
            dedup_ok=dedup_ok,
            contradiction_free=contradiction_free,
            requirements=requirements,
        ):
            return False
        with self.l3_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(validated_artifact, ensure_ascii=False) + "\n"
            )
        return True
