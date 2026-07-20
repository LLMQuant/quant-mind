"""Governance policy loader and runtime enforcement helpers."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quantmind.configs import RawFallbackNode
from quantmind.mind.memory import L3CommitRequirements, can_commit_to_l3


class GovernancePolicyError(ValueError):
    """Raised when governance policy content cannot be loaded or validated."""


class AgentGovernanceRule(BaseModel):
    """Tool + tier constraints for one role in a scenario."""

    model_config = ConfigDict(extra="forbid")

    role: str
    allowed_tools: list[str] = Field(default_factory=list)
    output_tier: Literal["L1", "L2", "L3"]
    required_schema: str | None = None
    confidence_threshold: float | None = None
    fallback_node: str | None = None


class ScenarioGovernanceRule(BaseModel):
    """Scenario-level set of role policies."""

    model_config = ConfigDict(extra="forbid")

    description: str
    agents: list[AgentGovernanceRule] = Field(default_factory=list)


class GlobalGovernanceSettings(BaseModel):
    """Global runtime limits and failure behavior."""

    model_config = ConfigDict(extra="forbid")

    loop_budget_max: int = 5
    fallback_policy: Literal["quarantine_and_continue", "fail_fast"] = (
        "quarantine_and_continue"
    )


class L3CommitPolicy(BaseModel):
    """Hard gates required before committing to durable memory.

    Mirrors :class:`quantmind.mind.memory.L3CommitRequirements` field for
    field so that :func:`enforce_l3_commit_gates` can delegate to the shared
    ``can_commit_to_l3`` helper. Keep these defaults in sync.
    """

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = 0.85
    require_provenance: bool = True
    require_schema_validity: bool = True
    require_dedup_check: bool = True
    require_contradiction_check: bool = True

    def to_requirements(self) -> L3CommitRequirements:
        """Convert to the dataclass used by ``mind.memory``."""
        return L3CommitRequirements(
            min_confidence=self.min_confidence,
            require_provenance=self.require_provenance,
            require_schema_validity=self.require_schema_validity,
            require_dedup_check=self.require_dedup_check,
            require_contradiction_check=self.require_contradiction_check,
        )


class GovernancePolicy(BaseModel):
    """Full governance policy document."""

    model_config = ConfigDict(extra="forbid")

    version: str
    global_settings: GlobalGovernanceSettings = Field(
        default_factory=GlobalGovernanceSettings
    )
    l3_commit: L3CommitPolicy = Field(default_factory=L3CommitPolicy)
    scenarios: dict[str, ScenarioGovernanceRule] = Field(default_factory=dict)

    def role_rule(
        self, scenario_name: str, role: str
    ) -> AgentGovernanceRule:
        """Resolve the role rule for one scenario."""
        try:
            scenario = self.scenarios[scenario_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown governance scenario: {scenario_name!r}"
            ) from error
        for rule in scenario.agents:
            if rule.role == role:
                return rule
        raise KeyError(
            f"Role {role!r} is not defined in scenario {scenario_name!r}"
        )


def load_governance_policy(path: str | Path | None = None) -> GovernancePolicy:
    """Load and validate governance policy YAML."""
    policy_path = Path(path) if path is not None else _default_policy_path()
    try:
        loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GovernancePolicyError(
            f"Unable to read governance policy: {policy_path}"
        ) from error
    except yaml.YAMLError as error:
        raise GovernancePolicyError(
            f"Invalid YAML in governance policy: {policy_path}"
        ) from error

    if loaded is None:
        raise GovernancePolicyError(
            f"Governance policy is empty: {policy_path}"
        )
    try:
        return GovernancePolicy.model_validate(loaded)
    except ValidationError as error:
        raise GovernancePolicyError(
            f"Governance policy schema validation failed: {policy_path}"
        ) from error


def ensure_tool_allowed(
    policy: GovernancePolicy,
    *,
    scenario_name: str,
    role: str,
    tool_name: str,
) -> None:
    """Raise when a role tries to use a tool outside its allowlist."""
    rule = policy.role_rule(scenario_name, role)
    if tool_name not in rule.allowed_tools:
        raise PermissionError(
            f"Tool {tool_name!r} is not allowed for role {role!r} "
            f"in scenario {scenario_name!r}"
        )


class LoopBudgetManager:
    """Track and enforce max loop budget from governance policy."""

    def __init__(self, max_loops: int) -> None:
        if max_loops <= 0:
            raise ValueError("max_loops must be positive")
        self.max_loops = max_loops
        self._consumed = 0

    @property
    def remaining(self) -> int:
        """Return remaining budget steps."""
        return self.max_loops - self._consumed

    def consume(self, *, steps: int = 1) -> int:
        """Consume loop budget and return the remaining amount."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self._consumed + steps > self.max_loops:
            raise RuntimeError(
                f"Loop budget exceeded: requested {steps}, "
                f"remaining {self.remaining}"
            )
        self._consumed += steps
        return self.remaining


def loop_budget_manager(policy: GovernancePolicy) -> LoopBudgetManager:
    """Create a budget manager from policy defaults."""
    return LoopBudgetManager(policy.global_settings.loop_budget_max)


def run_fallback_policy(
    policy: GovernancePolicy,
    *,
    target_schema: str,
    payload: dict[str, Any],
    error_message: str,
) -> RawFallbackNode:
    """Run fallback behavior configured by governance policy."""
    fallback = policy.global_settings.fallback_policy
    if fallback == "quarantine_and_continue":
        return RawFallbackNode(
            raw_payload=payload,
            target_schema=target_schema,
            error_message=error_message,
        )
    raise RuntimeError(
        f"Fallback policy {fallback!r} blocks continuation"
    )


def enforce_l3_commit_gates(
    policy: GovernancePolicy,
    *,
    artifact: dict[str, Any],
    schema_valid: bool,
    dedup_ok: bool,
    contradiction_free: bool,
) -> bool:
    """Return true only when all configured L3 commit gates pass.

    Delegates to :func:`quantmind.mind.memory.can_commit_to_l3` so there is
    a single source of truth for L3 commit gating semantics across flows
    and the mind/memory engine.
    """
    return can_commit_to_l3(
        artifact,
        schema_valid=schema_valid,
        dedup_ok=dedup_ok,
        contradiction_free=contradiction_free,
        requirements=policy.l3_commit.to_requirements(),
    )


def _default_policy_path() -> Path:
    return Path(__file__).with_name("governance.yaml")
