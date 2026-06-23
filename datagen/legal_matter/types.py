from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DATASET_VERSION = "legal-matter-chronology-v1"
PROMPT_VERSION = "2026-06-23-v1"
DEFAULT_MATTER_TYPE = "commercial_lease"
DEFAULT_MATTER_DOMAIN = "commercial_lease"
DEFAULT_JURISDICTION_STYLE = "jurisdiction-light Australian"
DEFAULT_MODEL = "openai/gpt-5.4"

MatterDomain = Literal[
    "commercial_lease",
    "employment_dispute",
    "family_property_settlement",
]

DOMAIN_CONFIGS: dict[str, dict[str, str]] = {
    "commercial_lease": {
        "label": "commercial lease dispute",
        "matter_id_prefix": "lease_matter",
        "setting": (
            "a tenant, landlord, property manager, solicitors, contractors, "
            "and at least one third-party witness or expert"
        ),
        "required_issues": (
            "at least one rent or outgoings issue, one repairs/access issue, "
            "one notice or deadline issue, and one source-authority issue such "
            "as draft vs final, recollection vs contemporaneous record, or "
            "offer vs accepted term"
        ),
    },
    "employment_dispute": {
        "label": "employment dispute",
        "matter_id_prefix": "employment_matter",
        "setting": (
            "an employee, employer, HR lead, direct manager, workplace "
            "investigator, solicitors, and at least one medical, payroll, or "
            "witness source"
        ),
        "required_issues": (
            "at least one performance or conduct allegation, one procedural "
            "fairness issue, one leave/medical/accommodation issue, one pay or "
            "termination entitlement issue, and one source-authority issue such "
            "as draft warning vs final letter, Slack recollection vs HR record, "
            "or unsigned settlement term vs accepted position"
        ),
    },
    "family_property_settlement": {
        "label": "family property settlement dispute",
        "matter_id_prefix": "family_property_matter",
        "setting": (
            "separating spouses or de facto partners, family lawyers, a "
            "financial adviser or accountant, a property valuer, a mediator, "
            "and at least one bank, employer, or business record source"
        ),
        "required_issues": (
            "at least one asset disclosure issue, one valuation or liquidity "
            "issue, one bank/transaction chronology issue, one interim payment "
            "or undertaking issue, and one source-authority issue such as draft "
            "asset schedule vs bank record, text message recollection vs formal "
            "letter, or preliminary valuation vs updated report"
        ),
    },
}


ArtifactType = Literal[
    "email_thread",
    "slack_thread",
    "legal_document",
    "client_note",
    "inspection_report",
    "ledger_csv",
    "photo_placeholder",
    "screenshot_placeholder",
]

ReliabilityLevel = Literal["high", "medium", "low"]
QuestionCategory = Literal[
    "chronology",
    "contradiction",
    "source_authority",
    "party_knowledge",
    "live_issues",
    "deadlines",
    "current_position",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Party(StrictBaseModel):
    party_id: str
    name: str
    role: str
    organisation: str | None = None
    notes: str | None = None


class ChronologyDraft(StrictBaseModel):
    title: str
    summary: str
    parties: list[Party] = Field(min_length=4)
    chronology_text: str


class CanonicalEvent(StrictBaseModel):
    matter_id: str = ""
    event_id: str
    sequence_index: int = Field(ge=1)
    relative_day: int = Field(ge=1)
    date: str | None = None
    title: str
    parties: list[str] = Field(default_factory=list)
    factual_assertions: list[str] = Field(default_factory=list)
    disputed_facts: list[str] = Field(default_factory=list)
    legal_or_commercial_significance: str
    later_impact: str
    material: bool = True


class Contradiction(StrictBaseModel):
    matter_id: str = ""
    contradiction_id: str
    description: str
    event_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    intentional: bool = True
    resolution: str | None = None


class MatterStateSnapshot(StrictBaseModel):
    matter_id: str = ""
    snapshot_id: str
    checkpoint_label: str
    relative_day: int = Field(ge=1)
    date: str | None = None
    live_issues: list[str] = Field(default_factory=list)
    resolved_issues: list[str] = Field(default_factory=list)
    disputed_facts: list[str] = Field(default_factory=list)
    obligations: list[str] = Field(default_factory=list)
    deadlines: list[str] = Field(default_factory=list)
    evidence_position: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    supporting_artifact_ids: list[str] = Field(default_factory=list)


class Matter(StrictBaseModel):
    matter_id: str
    title: str
    jurisdiction_style: str = DEFAULT_JURISDICTION_STYLE
    matter_type: str = DEFAULT_MATTER_TYPE
    parties: list[Party] = Field(default_factory=list)
    summary: str
    event_ids: list[str] = Field(default_factory=list)
    checkpoint_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_now_iso)


class StructuredMatter(StrictBaseModel):
    matter: Matter
    canonical_events: list[CanonicalEvent] = Field(min_length=8)
    contradictions: list[Contradiction] = Field(default_factory=list)
    state_snapshots: list[MatterStateSnapshot] = Field(min_length=3)


class ArtifactSpec(StrictBaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    title: str
    author: str
    recipients: list[str] = Field(default_factory=list)
    displayed_date: str
    date_format_hint: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    facts_revealed: list[str] = Field(default_factory=list)
    facts_distorted: list[str] = Field(default_factory=list)
    intended_ambiguity: str | None = None
    reliability: ReliabilityLevel = "medium"
    file_name: str

    @field_validator("file_name")
    @classmethod
    def file_name_must_be_relative(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("file_name must be a relative path inside the artifact folder")
        return value


class ArtifactBlueprint(StrictBaseModel):
    artifacts: list[ArtifactSpec] = Field(min_length=6)


class ArtifactContent(StrictBaseModel):
    artifact_id: str
    content: str
    text_summary: str
    visible_dates: list[str] = Field(default_factory=list)


class GeneratedArtifact(StrictBaseModel):
    artifact_id: str
    matter_id: str
    artifact_type: ArtifactType
    title: str
    file_path: str
    text_summary: str
    referenced_event_ids: list[str] = Field(default_factory=list)
    visible_dates: list[str] = Field(default_factory=list)
    reliability_level: ReliabilityLevel
    sha256: str


class GoldQuestion(StrictBaseModel):
    question_id: str
    matter_id: str
    category: QuestionCategory
    prompt: str
    expected_answer: str
    required_evidence_artifact_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    scoring_rubric: str


class GoldQuestionSet(StrictBaseModel):
    questions: list[GoldQuestion] = Field(min_length=6)


class GDPValTask(StrictBaseModel):
    task_id: str
    sector: str
    occupation: str
    domain: str
    matter_id: str
    matter_title: str
    category: QuestionCategory
    prompt: str
    reference_files: list[str] = Field(default_factory=list)
    reference_file_urls: list[str] = Field(default_factory=list)
    reference_file_hf_uris: list[str] = Field(default_factory=list)
    deliverable_text: str
    deliverable_files: list[str] = Field(default_factory=list)
    scoring_rubric: str
    source_event_ids: list[str] = Field(default_factory=list)
    gold_answer: str


class Manifest(StrictBaseModel):
    dataset_version: str = DATASET_VERSION
    prompt_version: str = PROMPT_VERSION
    generated_at: str = Field(default_factory=utc_now_iso)
    generator_model: str
    matter_type: str = DEFAULT_MATTER_TYPE
    jurisdiction_style: str = DEFAULT_JURISDICTION_STYLE
    count: int
    defaults: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(StrictBaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    matter_id: str | None = None
    artifact_id: str | None = None
    event_id: str | None = None
    snapshot_id: str | None = None


class ValidationReport(StrictBaseModel):
    dataset_dir: str
    checked_at: str = Field(default_factory=utc_now_iso)
    passed: bool
    issue_count: int
    issues: list[ValidationIssue] = Field(default_factory=list)
