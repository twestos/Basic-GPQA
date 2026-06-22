import asyncio
import json
import numpy as np
from eval.gpqa import GPQA


REPEAT = 5

async def main():
    all_results = []
    for i in range(REPEAT):
        with open(f"results_{i+1}.jsonl", "a") as f:
            print(f"Running repeat {i+1} of {REPEAT}")

            gdpqa = GPQA(model="openai/gpt-5.4-nano")
            gdpqa.load_dataset("diamond")
            results = await gdpqa.run()
            all_results.extend(results)

            # Perf metrics
            ttft_p50, ttft_p75, ttft_p90, ttft_p95, ttft_p99 = np.percentile([result.ttft for result in results], [50, 75, 90, 95, 99])
            tps_p50, tps_p75, tps_p90, tps_p95, tps_p99 = np.percentile([result.output_speed for result in results], [50, 75, 90, 95, 99])

            # Cost metrics
            total_cost = sum([result.cost for result in results])
            cost_per_question = total_cost / len(results)

            # Accuracy metrics
            pass_at_1 = sum(result.correctly_answered for result in results) / len(results)

            f.write(json.dumps({
                "repeat": i+1,
                "ttft_p50": ttft_p50,
                "ttft_p75": ttft_p75,
                "ttft_p90": ttft_p90,
                "ttft_p95": ttft_p95,
                "ttft_p99": ttft_p99,
                "tps_p50": tps_p50,
                "tps_p75": tps_p75,
                "tps_p90": tps_p90,
                "tps_p95": tps_p95,
                "tps_p99": tps_p99,
                "total_cost": total_cost,
                "cost_per_question": cost_per_question,
                "accuracy": pass_at_1,
                "pass_at_1": pass_at_1,
            }) + "\n")

    # Calculate the final score
    aggregate_score = sum(result.correctly_answered for result in all_results) / len(all_results)
    print(f"Aggregate pass@1: {aggregate_score:.4f}")

    with open("results_summary.jsonl", "a") as f:
        f.write(json.dumps({
            "repeats": REPEAT,
            "total_attempts": len(all_results),
            "correct_attempts": sum(result.correctly_answered for result in all_results),
            "pass_at_1": aggregate_score,
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
