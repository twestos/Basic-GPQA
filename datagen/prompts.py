BASE_CHRONOLOGY_PROMPT = """
Create a fictional but realistic chronology of events for this business context:

{business_context}

The chronology will be used as source material for an LLM evaluation. Its purpose
is to test whether a model can reason over a developing factual record where
events interact over time, people change their positions, and later facts may
clarify, complicate, or contradict earlier understandings.

Write the chronology as a sequence of dated narrative event entries. Each entry
should be self-contained enough that a later evaluator can extract:
- the key event or decision
- the parties involved and their roles
- the facts asserted, disputed, or newly discovered
- the practical, legal, commercial, personal, or strategic implications
- the impact on later events

Generation requirements:
- Use a realistic cast of named parties, advisors, counterparties, witnesses, and
  institutions appropriate to the business context.
- Include enough twists and turns to make the chronology non-trivial: competing
  accounts, mistaken assumptions, delayed disclosures, shifting incentives,
  ambiguous communications, changes in strategy, revised proposals, and facts
  that later turn out to be incomplete or misleading.
- Preserve temporal causality. Later entries should clearly build on earlier
  ones, revisit them, or change their significance.
- Include both explicit events, such as meetings, emails, filings, inspections,
  calls, approvals, payments, notices, or discoveries, and implicit developments,
  such as changed negotiating leverage, loss of trust, narrowing options, or new
  risks.
- Do not make every issue neat or resolved. Some facts should remain contested,
  some implications should be uncertain, and some parties should interpret the
  same event differently.
- Avoid generic filler. Every event should add a material fact, contradiction,
  implication, or consequence.
- Keep the scenario fictional. Do not use real private people, real confidential
  disputes, or identifiable ongoing matters.

Output requirements:
- Return a Chronology object with an `items` list.
- Each item must have:
  - `day`: an integer day number, starting at 1 and increasing over time.
  - `event`: a concise event title.
  - `details`: a rich narrative paragraph describing what happened, who was
    involved, what changed, what was disputed, and why the event matters.
- Produce 18 to 30 chronology items unless the context strongly requires a
  different number.
- Make the chronology dense enough that it can support later question generation
  about ordering, causation, contradictions, party knowledge, implications, and
  counterfactual reasoning.
""".strip()


def build_chronology_prompt(business_context: str) -> str:
    return BASE_CHRONOLOGY_PROMPT.format(business_context=business_context.strip())