import asyncio
import json
import numpy as np
from eval.gpqa import GPQA
from datetime import datetime


async def run_gpqa(repeat: int = 5, model: str = "google/gemini-3.1-flash-lite"):
    all_results = []
    for i in range(repeat):
        with open(f"results/results_{model.replace('/', '_')}.jsonl", "a") as f:
            print(f"Running repeat {i+1} of {repeat}")

            gdpqa = GPQA(model=model)
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
                "model": model,
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

    # Calculate the final score
    aggregate_score = sum(result.correctly_answered for result in all_results) / len(all_results)
    print(f"Aggregate pass@1: {aggregate_score:.4f}")

    with open(f"results_summary.jsonl", "a") as f:
        f.write(json.dumps({
            "repeats": repeat,
            "total_attempts": len(all_results),
            "correct_attempts": sum(result.correctly_answered for result in all_results),
            "pass_at_1": aggregate_score,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(run_gpqa())
