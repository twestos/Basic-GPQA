from __future__ import annotations

import json

from pydantic import BaseModel

from datagen.legal_matter.types import (
    DEFAULT_JURISDICTION_STYLE,
    DOMAIN_CONFIGS,
    MatterDomain,
)


def _dump(model: BaseModel | list[BaseModel] | dict) -> str:
    if isinstance(model, BaseModel):
        return model.model_dump_json(indent=2)
    if isinstance(model, list):
        return json.dumps([item.model_dump(mode="json") for item in model], indent=2)
    return json.dumps(model, indent=2)


CHRONOLOGY_INSTRUCTIONS = """
You create fictional legal matter chronologies for LLM evaluation datasets.
Use realistic commercial correspondence, evidence, and negotiation dynamics.
Do not use real private people, real confidential disputes, or identifiable
ongoing matters. Do not give legal advice. Keep the matter fictional.
""".strip()


STRUCTURE_INSTRUCTIONS = """
You convert fictional legal chronologies into canonical hidden state for an LLM
evaluation. Preserve temporal causality, mark contested facts, and distinguish
current matter state from obsolete earlier beliefs.
""".strip()


BLUEPRINT_INSTRUCTIONS = """
You design realistic legal matter artifacts that map to canonical events.
Artifacts should be messy enough to test chronology and source-authority
reasoning, but every ambiguity must be intentional and traceable.
""".strip()


ARTIFACT_INSTRUCTIONS = """
You write synthetic legal matter artifacts for an LLM benchmark. The artifact
must be self-contained and realistic, but fictional. Include the displayed date
exactly as specified in the artifact spec. Do not mention hidden event IDs.
""".strip()


QUESTION_INSTRUCTIONS = """
You write gold answer sets for chronology and current-position legal matter
evaluation. Each answer must be evidence-grounded and should reward careful
reasoning over timing, source authority, contradictions, and current state.
""".strip()


def build_chronology_prompt(matter_number: int, domain: MatterDomain) -> str:
    domain_config = DOMAIN_CONFIGS[domain]
    return f"""
Create a fictional but realistic chronology for one synthetic legal matter.

Matter defaults:
- Matter type: {domain_config["label"]}.
- Jurisdiction style: {DEFAULT_JURISDICTION_STYLE}.
- Business/legal setting: {domain_config["setting"]}.

The chronology should test whether a model can understand how the legal matter
changes over time. Include:
- 18 to 26 dated or date-like events.
- shifting negotiation positions and live issues.
- contradictory accounts that later evidence clarifies or complicates.
- {domain_config["required_issues"]}.
- enough parties to support emails, Slack-style internal messages, legal notes,
  CSV ledgers, and inspection/photo placeholder artifacts.

Return a ChronologyDraft object. This is matter number {matter_number}.
""".strip()


def build_structure_prompt(
    chronology: BaseModel,
    matter_id: str,
    domain: MatterDomain,
) -> str:
    domain_config = DOMAIN_CONFIGS[domain]
    return f"""
Convert this chronology into canonical hidden state for dataset matter
`{matter_id}`.

Requirements:
- Use stable IDs: event IDs like EVT-001, contradiction IDs like CON-001,
  snapshot IDs like STATE-001.
- Create 18 to 26 canonical events.
- Create 3 to 6 matter state snapshots at meaningful checkpoints.
- State snapshots must describe the matter as it stood at that checkpoint:
  live issues, resolved issues, disputed facts, obligations, deadlines,
  evidence position, and open questions.
- Mark material events with material=true.
- Set the Matter object's matter_type to "{domain}".
- Keep the legal matter type focused on {domain_config["label"]}.
- Keep the jurisdiction style as "{DEFAULT_JURISDICTION_STYLE}".
- Do not invent real legal citations or precise legal advice.

ChronologyDraft:
{_dump(chronology)}

Return a StructuredMatter object.
""".strip()


def build_blueprint_prompt(structured_matter: BaseModel) -> str:
    return f"""
Create an artifact blueprint for this synthetic legal matter.

Requirements:
- Produce 10 to 16 artifacts.
- Cover every material canonical event at least once.
- Include at least:
  - 3 email_thread artifacts.
  - 2 slack_thread artifacts.
  - 2 legal_document/client_note/inspection_report markdown artifacts.
  - 1 ledger_csv artifact.
  - 1 photo_placeholder or screenshot_placeholder markdown artifact.
- Vary displayed date formats and where dates appear.
- Some artifacts may contain mistaken, incomplete, or strategic statements, but
  every distortion must be listed in facts_distorted or intended_ambiguity.
- Use relative file names only, such as email_001_notice_chain.md or
  ledger_rent_arrears.csv.

StructuredMatter:
{_dump(structured_matter)}

Return an ArtifactBlueprint object.
""".strip()


def build_artifact_prompt(structured_matter: BaseModel, artifact_spec: BaseModel) -> str:
    return f"""
Write the full artifact content for this artifact spec.

Output rules:
- Return an ArtifactContent object.
- artifact_id must match the spec exactly.
- If artifact_type is ledger_csv, content must be valid CSV with a header row.
- For markdown artifacts, content should read like the specified artifact type.
- Do not expose hidden event IDs, source_event_ids, or validation notes.
- Include realistic but fictional names, amounts, dates, places, and document
  conventions from the matter.
- If the spec contains facts_distorted or intended_ambiguity, reflect that
  naturally in the artifact without explaining it as a benchmark device.

StructuredMatter:
{_dump(structured_matter)}

ArtifactSpec:
{_dump(artifact_spec)}
""".strip()


def build_questions_prompt(
    structured_matter: BaseModel,
    artifact_specs: list[BaseModel],
    matter_id: str,
) -> str:
    return f"""
Create gold benchmark questions for this synthetic legal matter.

Requirements:
- Return a GoldQuestionSet object.
- Create 8 to 12 questions.
- Include at least one question in each category:
  chronology, contradiction, source_authority, party_knowledge, live_issues,
  deadlines, current_position.
- Every question must have an evidence-grounded expected answer.
- required_evidence_artifact_ids must reference artifact IDs from the blueprint.
- source_event_ids must reference canonical event IDs from the structured matter.
- At least two questions should require combining 3 or more artifacts.
- Include at least one "where does the matter stand now?" style question.

Matter ID: {matter_id}

StructuredMatter:
{_dump(structured_matter)}

ArtifactSpecs:
{_dump(artifact_specs)}
""".strip()
