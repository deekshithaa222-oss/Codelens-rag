"""Batch evaluation runner for RAG systems"""
import json
from typing import List, Dict, Any
from pathlib import Path
from backend.logger import logger
from .scorer import FaithfulnessScorer


class BatchEvalRunner:
    """Runs batch evaluation on RAG results.
    
    Tradeoff: Batch eval is cheaper than interactive but doesn't catch
    all failure modes. For MVP, we evaluate on a manual gold set of Q&A.
    """

    def __init__(self, eval_dataset_path: str = "eval_set.json"):
        """Initialize evaluation runner.
        
        Args:
            eval_dataset_path: Path to evaluation dataset
        """
        self.eval_dataset_path = Path(eval_dataset_path)
        self.scorer = FaithfulnessScorer()
        self.results = []

    def load_eval_set(self) -> List[Dict[str, Any]]:
        """Load evaluation set from JSON.
        
        Returns:
            List of evaluation cases with 'question', 'context', 'expected_answer'
        """
        if not self.eval_dataset_path.exists():
            logger.warning(f"Eval set not found: {self.eval_dataset_path}")
            return []

        with open(self.eval_dataset_path) as f:
            return json.load(f)

    def evaluate(
        self,
        generated_responses: List[Dict[str, Any]],
        eval_set: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run evaluation on generated responses.
        
        Args:
            generated_responses: List of {question, response, context}
            eval_set: Optional evaluation set (uses default if None)
            
        Returns:
            Aggregated evaluation metrics
        """
        if eval_set is None:
            eval_set = self.load_eval_set()

        self.results = []
        scores = []

        for item in generated_responses:
            result = {
                "question": item.get("question"),
                "response": item.get("response"),
                "context": item.get("context", ""),
            }

            # Score faithfulness
            faithfulness = self.scorer.score(
                item["response"],
                item.get("context", ""),
                item.get("question")
            )
            result["faithfulness"] = faithfulness["score"]
            result["details"] = faithfulness

            self.results.append(result)
            scores.append(faithfulness["score"])

        # Compute aggregate metrics
        metrics = {
            "total_evals": len(self.results),
            "mean_faithfulness": sum(scores) / len(scores) if scores else 0,
            "min_faithfulness": min(scores) if scores else 0,
            "max_faithfulness": max(scores) if scores else 0,
            "pass_rate": sum(1 for s in scores if s > 0.7) / len(scores) if scores else 0,
            "results": self.results
        }

        return metrics

    def save_results(self, output_path: str = "eval_results.json") -> None:
        """Save evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved {len(self.results)} eval results to {output_path}")

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print(f"\\n{'='*50}")
        print("Evaluation Summary")
        print(f"{'='*50}")
        print(f"Total evaluations: {metrics['total_evals']}")
        print(f"Mean faithfulness: {metrics['mean_faithfulness']:.2%}")
        print(f"Faithfulness range: {metrics['min_faithfulness']:.2%} - {metrics['max_faithfulness']:.2%}")
        print(f"Pass rate (>70%): {metrics['pass_rate']:.2%}")
        print(f"{'='*50}\\n")
