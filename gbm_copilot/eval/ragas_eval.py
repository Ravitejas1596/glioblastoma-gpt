"""
Ragas Evaluation Script
Evaluates GBM Copilot on the golden question set.
Metrics: faithfulness, answer_relevancy, context_recall
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


async def evaluate_question(
    question: dict[str, Any],
    use_ragas: bool = True,
) -> dict[str, Any]:
    """Evaluate a single golden question through the full pipeline."""
    from gbm_copilot.agents.graph import run_query

    result = await run_query(
        query=question["question"],
        literacy_mode=question.get("literacy_mode", "patient"),
    )

    answer = result.get("final_answer", "")
    contexts = [c.get("text", "") for c in result.get("research_results", [])[:5]]
    citations = result.get("citations", [])

    # Keyword coverage check (simple accuracy proxy)
    keywords = question.get("expected_keywords", [])
    answer_lower = answer.lower()
    keyword_hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    keyword_coverage = keyword_hits / len(keywords) if keywords else 1.0

    # Source coverage check
    expected_sources = question.get("expected_sources", [])
    found_sources = {c.get("source", "") for c in citations}
    source_coverage = (
        len(set(expected_sources) & found_sources) / len(expected_sources)
        if expected_sources else 1.0
    )

    eval_result = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "answer": answer,
        "keyword_coverage": keyword_coverage,
        "source_coverage": source_coverage,
        "confidence_score": result.get("confidence_score", 0.0),
        "safety_flags": result.get("safety_flags", []),
        "is_blocked": result.get("is_blocked", False),
        "query_type": result.get("query_type", ""),
        "citation_count": len(citations),
    }

    # Ragas evaluation
    if use_ragas and contexts:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall
            from datasets import Dataset

            ragas_data = Dataset.from_dict({
                "question": [question["question"]],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [" ".join(keywords)],
            })

            ragas_result = evaluate(
                ragas_data,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )
            eval_result.update({
                "faithfulness": ragas_result["faithfulness"],
                "answer_relevancy": ragas_result["answer_relevancy"],
                "context_recall": ragas_result.get("context_recall", 0.0),
            })
        except Exception as e:
            console.print(f"  [yellow]Ragas eval failed for {question['id']}: {e}[/]")

    return eval_result


async def run_evaluation(
    golden_path: Path = GOLDEN_SET_PATH,
    output_path: Path | None = None,
    max_questions: int | None = None,
    use_ragas: bool = True,
) -> dict[str, Any]:
    """Run full evaluation on the golden set."""
    with open(golden_path) as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    console.print(f"\n[bold cyan]🧪 GlioblastomaGPT Evaluation[/]")
    console.print(f"Questions: {len(questions)} | Ragas: {'✓' if use_ragas else '✗'}\n")

    results = []
    for i, q in enumerate(questions, 1):
        console.print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        try:
            result = await evaluate_question(q, use_ragas=use_ragas)
            results.append(result)
            kw = result['keyword_coverage']
            color = "green" if kw > 0.7 else "yellow" if kw > 0.4 else "red"
            console.print(f"  Keyword coverage: [{color}]{kw:.1%}[/] | Confidence: {result['confidence_score']:.2f}")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/]")

    # Aggregate metrics
    agg = _aggregate_metrics(results)

    # Display table
    _display_results_table(results, agg)

    # Save results
    if output_path:
        output = {"results": results, "aggregate": agg}
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to {output_path}[/]")

    return {"results": results, "aggregate": agg}


def _aggregate_metrics(results: list[dict]) -> dict[str, float]:
    def safe_avg(key):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    by_category: dict[str, list] = {}
    for r in results:
        cat = r.get("category", "other")
        by_category.setdefault(cat, []).append(r.get("keyword_coverage", 0))

    return {
        "avg_keyword_coverage": safe_avg("keyword_coverage"),
        "avg_source_coverage": safe_avg("source_coverage"),
        "avg_confidence": safe_avg("confidence_score"),
        "avg_faithfulness": safe_avg("faithfulness"),
        "avg_answer_relevancy": safe_avg("answer_relevancy"),
        "avg_context_recall": safe_avg("context_recall"),
        "blocked_rate": sum(1 for r in results if r.get("is_blocked")) / len(results),
        "by_category": {cat: sum(scores)/len(scores) for cat, scores in by_category.items()},
    }


def _display_results_table(results: list[dict], agg: dict):
    table = Table(title="GBM Copilot Evaluation Results", style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("KW Cov.", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Blocked")

    for r in results:
        kw = r.get("keyword_coverage", 0)
        conf = r.get("confidence_score", 0)
        faith = r.get("faithfulness", "-")
        table.add_row(
            r["id"],
            r["category"],
            f"{'🟢' if kw > 0.7 else '🟡' if kw > 0.4 else '🔴'} {kw:.0%}",
            f"{conf:.2f}",
            f"{faith:.2f}" if isinstance(faith, float) else str(faith),
            "✗" if r.get("is_blocked") else "✓",
        )

    console.print(table)
    console.print(f"\n[bold]Aggregate Metrics:[/]")
    console.print(f"  Keyword Coverage: [green]{agg['avg_keyword_coverage']:.1%}[/]")
    console.print(f"  Source Coverage:  [green]{agg['avg_source_coverage']:.1%}[/]")
    console.print(f"  Avg Confidence:   [cyan]{agg['avg_confidence']:.2f}[/]")
    if agg.get("avg_faithfulness"):
        console.print(f"  Faithfulness:     [blue]{agg['avg_faithfulness']:.2f}[/]")
    console.print(f"  Blocked Rate:     {agg['blocked_rate']:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ragas Evaluation")
    parser.add_argument("--golden", type=str, default=str(GOLDEN_SET_PATH))
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        golden_path=Path(args.golden),
        output_path=Path(args.output),
        max_questions=args.max,
        use_ragas=not args.no_ragas,
    ))
