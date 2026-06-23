from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from datagen.client import DatasetGeneratorBase
from datagen.legal_matter.prompts import (
    ARTIFACT_INSTRUCTIONS,
    BLUEPRINT_INSTRUCTIONS,
    CHRONOLOGY_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    STRUCTURE_INSTRUCTIONS,
    build_artifact_prompt,
    build_blueprint_prompt,
    build_chronology_prompt,
    build_questions_prompt,
    build_structure_prompt,
)
from datagen.legal_matter.types import (
    ArtifactBlueprint,
    ArtifactContent,
    ArtifactSpec,
    ChronologyDraft,
    Contradiction,
    DEFAULT_JURISDICTION_STYLE,
    DEFAULT_MATTER_TYPE,
    DEFAULT_MODEL,
    GoldQuestion,
    GoldQuestionSet,
    Matter,
    MatterStateSnapshot,
    MatterDomain,
    Party,
    StructuredMatter,
    CanonicalEvent,
)
from datagen.types import DatagenClientConfig


OutputT = TypeVar("OutputT", bound=BaseModel)


class LegalMatterBackend(Protocol):
    def generate_chronology(
        self,
        matter_number: int,
        domain: MatterDomain,
    ) -> ChronologyDraft:
        ...

    def structure_matter(
        self,
        chronology: ChronologyDraft,
        matter_id: str,
        domain: MatterDomain,
    ) -> StructuredMatter:
        ...

    def create_artifact_blueprint(self, structured_matter: StructuredMatter) -> ArtifactBlueprint:
        ...

    def generate_artifact(
        self,
        structured_matter: StructuredMatter,
        artifact_spec: ArtifactSpec,
    ) -> ArtifactContent:
        ...

    def generate_questions(
        self,
        structured_matter: StructuredMatter,
        artifact_specs: list[ArtifactSpec],
        matter_id: str,
    ) -> GoldQuestionSet:
        ...


class OpenRouterLegalMatterBackend:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def _generate(
        self,
        output_type: type[OutputT],
        instructions: str,
        prompt: str,
    ) -> OutputT:
        config: DatagenClientConfig[Any] = DatagenClientConfig(
            model=self.model,  # type: ignore[arg-type]
            instructions=instructions,
            output_format=output_type,
            prompt=prompt,
        )
        agent = DatasetGeneratorBase[OutputT](config)
        return agent.generate(prompt)

    def generate_chronology(
        self,
        matter_number: int,
        domain: MatterDomain,
    ) -> ChronologyDraft:
        return self._generate(
            ChronologyDraft,
            CHRONOLOGY_INSTRUCTIONS,
            build_chronology_prompt(matter_number, domain),
        )

    def structure_matter(
        self,
        chronology: ChronologyDraft,
        matter_id: str,
        domain: MatterDomain,
    ) -> StructuredMatter:
        return self._generate(
            StructuredMatter,
            STRUCTURE_INSTRUCTIONS,
            build_structure_prompt(chronology, matter_id, domain),
        )

    def create_artifact_blueprint(self, structured_matter: StructuredMatter) -> ArtifactBlueprint:
        return self._generate(
            ArtifactBlueprint,
            BLUEPRINT_INSTRUCTIONS,
            build_blueprint_prompt(structured_matter),
        )

    def generate_artifact(
        self,
        structured_matter: StructuredMatter,
        artifact_spec: ArtifactSpec,
    ) -> ArtifactContent:
        return self._generate(
            ArtifactContent,
            ARTIFACT_INSTRUCTIONS,
            build_artifact_prompt(structured_matter, artifact_spec),
        )

    def generate_questions(
        self,
        structured_matter: StructuredMatter,
        artifact_specs: list[ArtifactSpec],
        matter_id: str,
    ) -> GoldQuestionSet:
        return self._generate(
            GoldQuestionSet,
            QUESTION_INSTRUCTIONS,
            build_questions_prompt(structured_matter, artifact_specs, matter_id),
        )


class FixtureLegalMatterBackend:
    """Deterministic backend for local smoke tests without API calls."""

    def generate_chronology(
        self,
        matter_number: int,
        domain: MatterDomain = "commercial_lease",
    ) -> ChronologyDraft:
        title_by_domain = {
            "commercial_lease": "Harbour Pantry lease dispute",
            "employment_dispute": "Northstar Labs dismissal dispute",
            "family_property_settlement": "Vargas-Lin property settlement",
        }
        summary_by_domain = {
            "commercial_lease": "A fictional commercial lease dispute involving water ingress, rent withholding, disputed outgoings, and a contested notice to remedy.",
            "employment_dispute": "A fictional employment dispute involving performance allegations, medical leave, workplace messages, pay records, and contested termination process.",
            "family_property_settlement": "A fictional family property settlement involving asset disclosure, business valuation, bank transfers, interim payments, and contested undertakings.",
        }
        parties = [
            Party(party_id="PTY-001", name="Harbour Pantry Pty Ltd", role="tenant"),
            Party(party_id="PTY-002", name="Northbank Holdings Pty Ltd", role="landlord"),
            Party(party_id="PTY-003", name="Mira Shah", role="tenant solicitor", organisation="Shah Legal"),
            Party(party_id="PTY-004", name="Elliot Price", role="landlord solicitor", organisation="Price & Dale"),
            Party(party_id="PTY-005", name="Rosa Nguyen", role="property manager", organisation="Civic Property"),
            Party(party_id="PTY-006", name="Keane Building Services", role="contractor"),
        ]
        chronology_text = """
Day 1: Harbour Pantry signs a short-form retail lease for Shop 3 at 16 Kellett Lane.
Day 8: The tenant emails the property manager about water ingress near the rear storeroom.
Day 12: The landlord says the issue was caused by the tenant's cool-room fitout.
Day 18: A contractor inspection finds a blocked roof drain and staining that predates fitout works.
Day 24: The tenant withholds part of the rent and disputes outgoings entries.
Day 29: The landlord issues a notice to remedy relying on an arrears figure that includes disputed charges.
Day 36: Solicitors exchange a without-prejudice proposal about rent abatement and access for repairs.
Day 43: The parties narrow the dispute to arrears, repair access, and whether the notice remains valid.
""".strip()
        return ChronologyDraft(
            title=f"{title_by_domain[domain]} {matter_number}",
            summary=summary_by_domain[domain],
            parties=parties,
            chronology_text=chronology_text,
        )

    def structure_matter(
        self,
        chronology: ChronologyDraft,
        matter_id: str,
        domain: MatterDomain = "commercial_lease",
    ) -> StructuredMatter:
        events = [
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-001",
                sequence_index=1,
                relative_day=1,
                date="2026-02-03",
                title="Lease signed",
                parties=["PTY-001", "PTY-002"],
                factual_assertions=["The parties signed a short-form lease for Shop 3 at 16 Kellett Lane."],
                legal_or_commercial_significance="Establishes the lease relationship and baseline obligations.",
                later_impact="The executed lease becomes the source of authority over later informal statements.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-002",
                sequence_index=2,
                relative_day=8,
                date="2026-02-10",
                title="Tenant reports water ingress",
                parties=["PTY-001", "PTY-005"],
                factual_assertions=["The tenant reported water ingress near the rear storeroom."],
                legal_or_commercial_significance="Creates a contemporaneous notice of the repair issue.",
                later_impact="Later arguments turn on when the landlord first knew of the leak.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-003",
                sequence_index=3,
                relative_day=12,
                date="2026-02-14",
                title="Landlord blames fitout",
                parties=["PTY-002", "PTY-005", "PTY-001"],
                factual_assertions=["The landlord asserted the water issue was caused by tenant fitout works."],
                disputed_facts=["Whether the leak was caused by the tenant's cool-room installation."],
                legal_or_commercial_significance="Introduces the main factual contradiction about responsibility for repairs.",
                later_impact="The assertion is weakened by the later inspection report.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-004",
                sequence_index=4,
                relative_day=18,
                date="2026-02-20",
                title="Inspection finds pre-existing drain issue",
                parties=["PTY-006", "PTY-005"],
                factual_assertions=["The contractor observed a blocked roof drain and staining that appeared older than the fitout."],
                disputed_facts=["Whether the staining conclusively predates the tenant's occupation."],
                legal_or_commercial_significance="Provides higher-quality evidence than the landlord's earlier assertion.",
                later_impact="Shifts the repair-access dispute and weakens the landlord's notice position.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-005",
                sequence_index=5,
                relative_day=24,
                date="2026-02-26",
                title="Tenant withholds rent and disputes outgoings",
                parties=["PTY-001", "PTY-002"],
                factual_assertions=["The tenant withheld part of the rent and challenged two outgoings entries."],
                disputed_facts=["The correct arrears amount after excluding disputed outgoings."],
                legal_or_commercial_significance="Creates a live arrears issue tied to the repair dispute.",
                later_impact="The arrears ledger becomes important to whether the notice amount was reliable.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-006",
                sequence_index=6,
                relative_day=29,
                date="2026-03-03",
                title="Notice to remedy issued",
                parties=["PTY-002", "PTY-001", "PTY-004"],
                factual_assertions=["The landlord issued a notice to remedy using an arrears figure that included disputed charges."],
                disputed_facts=["Whether the notice amount overstated the tenant's arrears."],
                legal_or_commercial_significance="Creates a deadline and a source-authority issue about the notice.",
                later_impact="The current position depends on whether the notice is treated as defective or merely disputed.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-007",
                sequence_index=7,
                relative_day=36,
                date="2026-03-10",
                title="Settlement proposal exchanged",
                parties=["PTY-003", "PTY-004"],
                factual_assertions=["Solicitors exchanged a proposal for rent abatement and staged access for repair works."],
                disputed_facts=["Whether the landlord's response accepted the tenant's access conditions."],
                legal_or_commercial_significance="Shows negotiation movement but does not finally resolve the matter.",
                later_impact="The proposal is a tempting but unreliable source for current obligations.",
            ),
            CanonicalEvent(
                matter_id=matter_id,
                event_id="EVT-008",
                sequence_index=8,
                relative_day=43,
                date="2026-03-17",
                title="Current issues narrowed",
                parties=["PTY-001", "PTY-002", "PTY-003", "PTY-004"],
                factual_assertions=["The parties narrowed the dispute to arrears, repair access, and the notice's status."],
                legal_or_commercial_significance="Defines the current matter position.",
                later_impact="Gold current-position questions should be answered from this state rather than earlier allegations.",
            ),
        ]
        contradictions = [
            Contradiction(
                matter_id=matter_id,
                contradiction_id="CON-001",
                description="The landlord's fitout-causation assertion is contradicted by the contractor's older staining observation.",
                event_ids=["EVT-003", "EVT-004"],
                intentional=True,
                resolution="The contractor report is more contemporaneous technical evidence, but not conclusive.",
            ),
            Contradiction(
                matter_id=matter_id,
                contradiction_id="CON-002",
                description="The notice arrears figure includes outgoings entries that the tenant had already disputed.",
                event_ids=["EVT-005", "EVT-006"],
                intentional=True,
                resolution="The amount remains disputed and affects the notice's reliability.",
            ),
        ]
        snapshots = [
            MatterStateSnapshot(
                matter_id=matter_id,
                snapshot_id="STATE-001",
                checkpoint_label="Initial repair complaint",
                relative_day=12,
                date="2026-02-14",
                live_issues=["Water ingress and responsibility for investigation."],
                disputed_facts=["Whether tenant fitout caused the leak."],
                obligations=["Landlord/property manager to respond to reported water ingress."],
                evidence_position=["Tenant email is the main contemporaneous record."],
                supporting_event_ids=["EVT-001", "EVT-002", "EVT-003"],
            ),
            MatterStateSnapshot(
                matter_id=matter_id,
                snapshot_id="STATE-002",
                checkpoint_label="After inspection and arrears dispute",
                relative_day=29,
                date="2026-03-03",
                live_issues=["Repair responsibility.", "Correct arrears amount.", "Validity and effect of notice."],
                disputed_facts=["Cause of water ingress.", "Whether disputed outgoings should be counted as arrears."],
                obligations=["Tenant to respond to notice deadline.", "Landlord to coordinate repair access."],
                deadlines=["Notice deadline running from 2026-03-03."],
                evidence_position=["Inspection report is stronger evidence than the earlier landlord assertion."],
                supporting_event_ids=["EVT-004", "EVT-005", "EVT-006"],
            ),
            MatterStateSnapshot(
                matter_id=matter_id,
                snapshot_id="STATE-003",
                checkpoint_label="Current position",
                relative_day=43,
                date="2026-03-17",
                live_issues=["Arrears calculation.", "Repair access timetable.", "Whether notice remains valid."],
                resolved_issues=["The matter is no longer about whether water ingress occurred."],
                disputed_facts=["The exact arrears amount.", "Whether the landlord accepted access conditions."],
                obligations=["Parties need to confirm access dates and corrected arrears before escalation."],
                deadlines=["Notice issue remains live until withdrawn, replaced, or resolved."],
                evidence_position=["Executed lease, inspection report, ledger, notice, and solicitor correspondence are all required."],
                open_questions=["Will the landlord withdraw or amend the notice?", "What arrears figure is accepted after disputed outgoings are removed?"],
                supporting_event_ids=["EVT-004", "EVT-005", "EVT-006", "EVT-007", "EVT-008"],
            ),
        ]
        matter = Matter(
            matter_id=matter_id,
            title=chronology.title,
            jurisdiction_style=DEFAULT_JURISDICTION_STYLE,
            matter_type=domain,
            parties=chronology.parties,
            summary=chronology.summary,
            event_ids=[event.event_id for event in events],
            checkpoint_ids=[snapshot.snapshot_id for snapshot in snapshots],
        )
        return StructuredMatter(
            matter=matter,
            canonical_events=events,
            contradictions=contradictions,
            state_snapshots=snapshots,
        )

    def create_artifact_blueprint(self, structured_matter: StructuredMatter) -> ArtifactBlueprint:
        artifacts = [
            ArtifactSpec(
                artifact_id="ART-001",
                artifact_type="email_thread",
                title="Lease execution and access email chain",
                author="Rosa Nguyen",
                recipients=["Harbour Pantry Pty Ltd", "Northbank Holdings Pty Ltd"],
                displayed_date="3 Feb 2026",
                date_format_hint="day month year",
                source_event_ids=["EVT-001", "EVT-002"],
                facts_revealed=["Executed lease exists.", "Tenant reported water ingress within the first fortnight."],
                reliability="high",
                file_name="email_001_lease_and_leak.md",
            ),
            ArtifactSpec(
                artifact_id="ART-002",
                artifact_type="slack_thread",
                title="Property team discussion blaming fitout",
                author="Rosa Nguyen",
                recipients=["Civic Property leasing team"],
                displayed_date="14/02/26",
                date_format_hint="Australian numeric date",
                source_event_ids=["EVT-003"],
                facts_revealed=["Landlord side initially blamed the cool-room fitout."],
                facts_distorted=["Treats fitout causation as firmer than the evidence supported."],
                intended_ambiguity="Internal shorthand may overstate the landlord's actual evidence.",
                reliability="medium",
                file_name="slack_002_fitout_theory.md",
            ),
            ArtifactSpec(
                artifact_id="ART-003",
                artifact_type="inspection_report",
                title="Keane Building Services inspection note",
                author="Keane Building Services",
                recipients=["Civic Property"],
                displayed_date="20 February 2026",
                source_event_ids=["EVT-004"],
                facts_revealed=["Blocked roof drain and older staining observed."],
                intended_ambiguity="Report says staining appeared older but avoids a conclusive causation opinion.",
                reliability="high",
                file_name="inspection_003_keane_report.md",
            ),
            ArtifactSpec(
                artifact_id="ART-004",
                artifact_type="ledger_csv",
                title="March arrears and outgoings ledger",
                author="Northbank Holdings Pty Ltd",
                recipients=["Price & Dale", "Shah Legal"],
                displayed_date="2026-03-01",
                source_event_ids=["EVT-005", "EVT-006"],
                facts_revealed=["Arrears figure includes two disputed outgoings entries."],
                facts_distorted=["Ledger labels disputed outgoings as due without qualification."],
                reliability="medium",
                file_name="ledger_004_arrears.csv",
            ),
            ArtifactSpec(
                artifact_id="ART-005",
                artifact_type="legal_document",
                title="Notice to remedy breach",
                author="Price & Dale",
                recipients=["Harbour Pantry Pty Ltd", "Shah Legal"],
                displayed_date="3 March 2026 at 4.18 pm",
                source_event_ids=["EVT-006"],
                facts_revealed=["Notice relies on the higher arrears figure."],
                intended_ambiguity="The notice is formal, but the underlying arrears amount is contested.",
                reliability="high",
                file_name="notice_005_remedy_breach.md",
            ),
            ArtifactSpec(
                artifact_id="ART-006",
                artifact_type="email_thread",
                title="Without prejudice repair access proposal",
                author="Mira Shah",
                recipients=["Elliot Price"],
                displayed_date="Wed 10 Mar, 11:06",
                source_event_ids=["EVT-007", "EVT-008"],
                facts_revealed=["Solicitors discussed rent abatement and staged access but did not fully resolve the dispute."],
                intended_ambiguity="The response sounds cooperative but does not clearly accept every access condition.",
                reliability="medium",
                file_name="email_006_wp_access_proposal.md",
            ),
            ArtifactSpec(
                artifact_id="ART-007",
                artifact_type="photo_placeholder",
                title="Rear storeroom staining photo placeholder",
                author="Harbour Pantry Pty Ltd",
                recipients=["Shah Legal"],
                displayed_date="saved as IMG_4312, no visible date",
                source_event_ids=["EVT-002", "EVT-004"],
                facts_revealed=["Photo depicts staining near the rear storeroom."],
                intended_ambiguity="The file name has no reliable date metadata in the visible artifact.",
                reliability="low",
                file_name="photo_007_rear_storeroom_placeholder.md",
            ),
        ]
        return ArtifactBlueprint(artifacts=artifacts)

    def generate_artifact(
        self,
        structured_matter: StructuredMatter,
        artifact_spec: ArtifactSpec,
    ) -> ArtifactContent:
        content = self._fixture_content(artifact_spec)
        return ArtifactContent(
            artifact_id=artifact_spec.artifact_id,
            content=content,
            text_summary="; ".join(artifact_spec.facts_revealed),
            visible_dates=[artifact_spec.displayed_date],
        )

    def generate_questions(
        self,
        structured_matter: StructuredMatter,
        artifact_specs: list[ArtifactSpec],
        matter_id: str,
    ) -> GoldQuestionSet:
        questions = [
            GoldQuestion(
                question_id="Q-001",
                matter_id=matter_id,
                category="chronology",
                prompt="What happened before the landlord issued the notice that affected the reliability of the arrears amount?",
                expected_answer="The tenant had already disputed outgoings entries and the ledger still included them in the arrears figure.",
                required_evidence_artifact_ids=["ART-004", "ART-005"],
                source_event_ids=["EVT-005", "EVT-006"],
                scoring_rubric="Answer must identify the pre-notice outgoings dispute and link it to the notice amount.",
            ),
            GoldQuestion(
                question_id="Q-002",
                matter_id=matter_id,
                category="contradiction",
                prompt="Which later evidence complicated the landlord's claim that the leak was caused by tenant fitout works?",
                expected_answer="The Keane Building Services inspection note recorded a blocked roof drain and staining that appeared older than the fitout.",
                required_evidence_artifact_ids=["ART-002", "ART-003"],
                source_event_ids=["EVT-003", "EVT-004"],
                scoring_rubric="Answer must compare the internal fitout theory with the contractor inspection.",
            ),
            GoldQuestion(
                question_id="Q-003",
                matter_id=matter_id,
                category="source_authority",
                prompt="Why is the formal notice not enough by itself to establish the current arrears position?",
                expected_answer="The notice is formal, but it relies on a ledger amount that includes disputed outgoings, so the amount remains contested.",
                required_evidence_artifact_ids=["ART-004", "ART-005"],
                source_event_ids=["EVT-005", "EVT-006"],
                scoring_rubric="Answer must distinguish document formality from reliability of the underlying amount.",
            ),
            GoldQuestion(
                question_id="Q-004",
                matter_id=matter_id,
                category="party_knowledge",
                prompt="What did the property manager know before the fitout-causation theory appeared internally?",
                expected_answer="The property manager had already received the tenant's report of water ingress near the rear storeroom.",
                required_evidence_artifact_ids=["ART-001", "ART-002"],
                source_event_ids=["EVT-002", "EVT-003"],
                scoring_rubric="Answer must track knowledge across the tenant email and later internal thread.",
            ),
            GoldQuestion(
                question_id="Q-005",
                matter_id=matter_id,
                category="live_issues",
                prompt="Where does the matter stand now?",
                expected_answer="The current live issues are the corrected arrears amount, repair access timetable, and whether the notice remains valid or should be withdrawn or amended.",
                required_evidence_artifact_ids=["ART-003", "ART-004", "ART-005", "ART-006"],
                source_event_ids=["EVT-004", "EVT-005", "EVT-006", "EVT-007", "EVT-008"],
                scoring_rubric="Answer must state current issues, not just recount earlier allegations.",
            ),
            GoldQuestion(
                question_id="Q-006",
                matter_id=matter_id,
                category="deadlines",
                prompt="Which artifact created the key deadline pressure, and why is that deadline disputed?",
                expected_answer="The notice to remedy created the deadline pressure, and it is disputed because the arrears figure was based on a contested ledger.",
                required_evidence_artifact_ids=["ART-004", "ART-005"],
                source_event_ids=["EVT-005", "EVT-006"],
                scoring_rubric="Answer must identify both the notice and the reason the deadline/amount is contested.",
            ),
            GoldQuestion(
                question_id="Q-007",
                matter_id=matter_id,
                category="current_position",
                prompt="Which artifacts are needed to assess the current position rather than relying on the earliest complaint alone?",
                expected_answer="The inspection report, arrears ledger, notice, and solicitor access proposal are needed because they update repair responsibility, arrears, notice status, and negotiation position.",
                required_evidence_artifact_ids=["ART-003", "ART-004", "ART-005", "ART-006"],
                source_event_ids=["EVT-004", "EVT-005", "EVT-006", "EVT-007", "EVT-008"],
                scoring_rubric="Answer must explain why later artifacts change the matter state.",
            ),
        ]
        return GoldQuestionSet(questions=questions)

    def _fixture_content(self, spec: ArtifactSpec) -> str:
        if spec.artifact_type == "ledger_csv":
            return "\n".join(
                [
                    "date,item,amount,status,notes",
                    "2026-02-01,Base rent,8200.00,due,February rent",
                    "2026-02-24,Electrical outgoings true-up,1130.00,disputed,Tenant queried calculation",
                    "2026-02-24,Common area cleaning,640.00,disputed,Tenant says not supported by lease schedule",
                    "2026-03-01,Partial payment,-5000.00,received,Tenant paid without admission",
                    "2026-03-03,Notice amount,4970.00,claimed,Includes disputed outgoings",
                ]
            )

        if spec.artifact_type == "slack_thread":
            return f"""# {spec.title}

**Displayed date:** {spec.displayed_date}

Rosa Nguyen: Northbank thinks the cool-room fitout is the likely cause. Can we tell the tenant to get their installer back?

Leasing: Maybe, but do we have anything from Keane yet?

Rosa Nguyen: Not yet. I'm noting it as fitout-related for now, subject to inspection.
"""

        if spec.artifact_type == "inspection_report":
            return f"""# {spec.title}

Inspection date: {spec.displayed_date}

Keane Building Services attended Shop 3, 16 Kellett Lane. Water staining was visible around the rear storeroom bulkhead. The roof drain above the rear wall was blocked with leaf matter and debris. The staining pattern appeared older than the recent cool-room works, although no destructive testing was carried out.

Recommended next step: clear the drain, test during the next rain event, and arrange access for ceiling inspection.
"""

        if spec.artifact_type == "legal_document":
            return f"""# {spec.title}

Issued: {spec.displayed_date}

Price & Dale acts for Northbank Holdings Pty Ltd. Our client alleges Harbour Pantry Pty Ltd is in arrears in the amount of $4,970.00 and requires remedy within the period stated in the lease.

This notice relies on the landlord's arrears ledger current at 1 March 2026. Northbank reserves all rights.
"""

        if spec.artifact_type == "photo_placeholder":
            return f"""# {spec.title}

Visible file label: {spec.displayed_date}

Placeholder for future generated image artifact. Description: close-range phone photo of brown water staining on the rear storeroom ceiling above stacked dry goods. No visible timestamp appears in the image itself.
"""

        return f"""# {spec.title}

Date shown: {spec.displayed_date}
From: {spec.author}
To: {", ".join(spec.recipients)}

This fictional matter artifact records the following:

{chr(10).join(f"- {fact}" for fact in spec.facts_revealed)}

Please confirm whether your client will agree to access dates without prejudice to the disputed arrears and repair responsibility issues.
"""
