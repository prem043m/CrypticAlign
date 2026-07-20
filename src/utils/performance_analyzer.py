from datetime import datetime
from pathlib import Path

import pandas as pd


class PerformanceAnalyzer:
    """
    nextmatchAi Performance Analyzer

    Tracks:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC AUC
    - Recommendation Quality Improvements

    Generates:
    - Markdown Report
    - CSV History
    """

    def __init__(self):

        self.metrics_history = []

    def record_metrics(
        self,
        iteration,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        baseline_acceptance_rate=None,
        reranked_acceptance_rate=None
    ):

        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,

            "baseline_acceptance_rate":
                baseline_acceptance_rate,

            "reranked_acceptance_rate":
                reranked_acceptance_rate
        }

        self.metrics_history.append(record)

    def calculate_model_improvement(self):

        if len(self.metrics_history) < 2:
            return None

        baseline = self.metrics_history[0]
        latest = self.metrics_history[-1]

        return {
            "baseline_accuracy":
                baseline["accuracy"],

            "latest_accuracy":
                latest["accuracy"],

            "percentage_point_gain":
                latest["accuracy"]
                - baseline["accuracy"],

            "relative_improvement":
                (
                    (
                        latest["accuracy"]
                        - baseline["accuracy"]
                    )
                    /
                    baseline["accuracy"]
                ) * 100
        }

    def calculate_feedback_impact(self):

        if not self.metrics_history:
            return None

        latest = self.metrics_history[-1]

        baseline_rate = latest.get(
            "baseline_acceptance_rate"
        )

        reranked_rate = latest.get(
            "reranked_acceptance_rate"
        )

        if (
            baseline_rate is None
            or reranked_rate is None
        ):
            return None

        return {
            "baseline_acceptance_rate":
                baseline_rate,

            "reranked_acceptance_rate":
                reranked_rate,

            "percentage_point_gain":
                reranked_rate
                - baseline_rate,

            "relative_improvement":
                (
                    (
                        reranked_rate
                        - baseline_rate
                    )
                    /
                    baseline_rate
                ) * 100
        }

    def to_dataframe(self):

        return pd.DataFrame(
            self.metrics_history
        )

    def save_csv(
        self,
        filepath="reports/metrics_history.csv"
    ):

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.to_dataframe().to_csv(
            filepath,
            index=False
        )

        return filepath

    def generate_report(self):

        report = []

        report.append(
            "# nextmatchAi Performance Analysis Report\n"
        )

        report.append(
            f"Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        report.append(
            "## Model Evaluation History\n"
        )

        if self.metrics_history:
            headers = [
                "Iteration", "Timestamp", "Accuracy", "Precision",
                "Recall", "F1 Score", "ROC AUC",
                "Baseline Accept Rate", "Reranked Accept Rate"
            ]
            report.append("| " + " | ".join(headers) + " |")
            report.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for record in self.metrics_history:
                bar_rate = record.get("baseline_acceptance_rate")
                rer_rate = record.get("reranked_acceptance_rate")
                row = [
                    str(record["iteration"]),
                    str(record["timestamp"]),
                    f"{record['accuracy']:.4f}",
                    f"{record['precision']:.4f}",
                    f"{record['recall']:.4f}",
                    f"{record['f1_score']:.4f}",
                    f"{record['roc_auc']:.4f}",
                    f"{bar_rate:.2%}" if bar_rate is not None else "N/A",
                    f"{rer_rate:.2%}" if rer_rate is not None else "N/A"
                ]
                report.append("| " + " | ".join(row) + " |")
            report.append("")

        model_improvement = (
            self.calculate_model_improvement()
        )

        if model_improvement:

            report.append(
                "\n## Model Improvement Analysis\n"
            )

            report.append(
                f"- Baseline Accuracy: "
                f"{model_improvement['baseline_accuracy']:.2%}"
            )

            report.append(
                f"- Latest Accuracy: "
                f"{model_improvement['latest_accuracy']:.2%}"
            )

            report.append(
                f"- Percentage Point Gain: "
                f"{model_improvement['percentage_point_gain']:.2%}"
            )

            report.append(
                f"- Relative Improvement: "
                f"{model_improvement['relative_improvement']:.2f}%"
            )

        feedback_impact = (
            self.calculate_feedback_impact()
        )

        if feedback_impact:

            report.append(
                "\n## Feedback Learning Impact\n"
            )

            report.append(
                f"- Hybrid Acceptance Rate: "
                f"{feedback_impact['baseline_acceptance_rate']:.2%}"
            )

            report.append(
                f"- ML Re-Ranked Acceptance Rate: "
                f"{feedback_impact['reranked_acceptance_rate']:.2%}"
            )

            report.append(
                f"- Gain: "
                f"{feedback_impact['percentage_point_gain']:.2%}"
            )

            report.append(
                f"- Relative Improvement: "
                f"{feedback_impact['relative_improvement']:.2f}%"
            )

        if self.metrics_history:

            latest = self.metrics_history[-1]

            report.append(
                "\n## Current Production Metrics\n"
            )

            report.append(
                f"- Accuracy: "
                f"{latest['accuracy']:.4f}"
            )

            report.append(
                f"- Precision: "
                f"{latest['precision']:.4f}"
            )

            report.append(
                f"- Recall: "
                f"{latest['recall']:.4f}"
            )

            report.append(
                f"- F1 Score: "
                f"{latest['f1_score']:.4f}"
            )

            report.append(
                f"- ROC AUC: "
                f"{latest['roc_auc']:.4f}"
            )

        report.append(
            "\n## Conclusion\n"
        )

        report.append(
            "The nextmatchAi recommendation engine "
            "combines hybrid compatibility scoring "
            "with machine-learning feedback learning. "
            "The ML re-ranking stage improves recommendation "
            "quality by prioritizing candidates who are "
            "both compatible and likely to receive positive "
            "user feedback."
        )

        return "\n".join(report)

    def save_report(
        self,
        filepath="reports/performance_report.md"
    ):

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        report = self.generate_report()

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)

        return filepath