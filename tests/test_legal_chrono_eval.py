from __future__ import annotations

import unittest

from eval.legal_chrono import (
    build_answer_prompt,
    parse_grader_score,
    summarize_results,
)
from eval.types import (
    LegalChronoEvaluationResult,
    LegalChronoTask,
    ReferenceFileContent,
)


class LegalChronoEvalTests(unittest.TestCase):
    def test_parse_grader_score_plain_text(self) -> None:
        score = parse_grader_score(
            "Score: 4\nRationale: Mostly correct but missed one caveat."
        )

        self.assertTrue(score.valid_score)
        self.assertEqual(score.score, 4)
        self.assertEqual(score.normalized_score, 0.8)
        self.assertTrue(score.passed)
        self.assertEqual(score.rationale, "Mostly correct but missed one caveat.")

    def test_parse_grader_score_fallback_fraction(self) -> None:
        score = parse_grader_score("I would give this 3/5 because timing is confused.")

        self.assertTrue(score.valid_score)
        self.assertEqual(score.score, 3)
        self.assertFalse(score.passed)

    def test_build_answer_prompt_includes_references(self) -> None:
        task = _task()
        prompt = build_answer_prompt(
            task,
            [
                ReferenceFileContent(
                    path="reference_files/matter/email.md",
                    content="Email body here.",
                    truncated=False,
                )
            ],
        )

        self.assertIn("Question?", prompt)
        self.assertIn("reference_files/matter/email.md", prompt)
        self.assertIn("Email body here.", prompt)

    def test_summarize_results(self) -> None:
        results = [_result("Q1", 5), _result("Q2", 3)]

        summary = summarize_results(results)

        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["average_score"], 4)
        self.assertEqual(summary["pass_at_4"], 0.5)
        self.assertEqual(summary["total_cost"], 0.06)


def _task() -> LegalChronoTask:
    return LegalChronoTask(
        task_id="Q1",
        sector="Professional",
        occupation="Lawyer",
        domain="commercial_lease",
        matter_id="matter_1",
        matter_title="Matter",
        category="chronology",
        prompt="Question?",
        reference_files=["reference_files/matter/email.md"],
        reference_file_urls=[],
        reference_file_hf_uris=[],
        deliverable_text="Gold",
        deliverable_files=[],
        scoring_rubric="Rubric",
        source_event_ids=["EVT-001"],
        gold_answer="Gold",
    )


def _result(task_id: str, score: int) -> LegalChronoEvaluationResult:
    return LegalChronoEvaluationResult(
        task_id=task_id,
        domain="commercial_lease",
        matter_id="matter_1",
        category="chronology",
        occupation="Lawyer",
        prompt="Prompt",
        reference_files=[],
        reference_file_hf_uris=[],
        reference_files_truncated=[],
        gold_answer="Gold",
        scoring_rubric="Rubric",
        model="model",
        grader_model="grader",
        model_response="Answer",
        grader_response=f"Score: {score}\nRationale: ok",
        valid_score=True,
        score=score,
        normalized_score=score / 5,
        passed=score >= 4,
        grader_rationale="ok",
        answer_ttft=0.1,
        answer_output_speed=2.0,
        answer_input_tokens_client=10,
        answer_input_tokens_model=11,
        answer_output_tokens_client=12,
        answer_output_tokens_model=13,
        answer_cost=0.01,
        grader_ttft=0.2,
        grader_output_speed=3.0,
        grader_input_tokens_client=14,
        grader_input_tokens_model=15,
        grader_output_tokens_client=16,
        grader_output_tokens_model=17,
        grader_cost=0.02,
        total_cost=0.03,
    )


if __name__ == "__main__":
    unittest.main()
