# response_models.py
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel

# The metrics you’re using
MetricName = Literal["picking_sides", "mirroring", "who_said", "delusion"]


class LLMResponse(BaseModel):
    """
    One model's reply at one round, cleaned + validated.
    """
    model_name: str
    metric: MetricName
    round_index: int  # 1-based

    system_prompt: str
    user_prompt: str

    raw_output: str          # cleaned full text
    parsed_output: Any       # after parsing_func(raw_output)

    sycophancy_scores: Optional[Dict[str, float]] = None

    # flag to indicate if parsed_output looks valid for this metric
    is_parsed_valid: bool = True

    @classmethod
    def from_metric(
        cls,
        *,
        model_name: str,
        metric: str,
        round_index: int,
        system_prompt: str,
        user_prompt: str,
        raw_output: str,
        parsed_output: Any,
        sycophancy_scores: Optional[Dict[str, float]] = None,
    ) -> "LLMResponse":
        """
        - Clean raw_output (strip, truncate).
        - Normalize / validate parsed_output depending on metric.
        """
        # 1) Clean raw text (handles artifacts like extra spaces, huge blobs)
        raw_clean = (raw_output or "").strip()
        # Hard truncation guard, so artifacts don’t explode logs

        # 2) Normalize + validate parsed_output
        metric_norm = metric.lower()
        allowed_labels = None

        if metric_norm == "picking_sides":
            allowed_labels = {"me", "my friend"}
        elif metric_norm == "mirroring":
            allowed_labels = {"yes", "no"}
        elif metric_norm in {"who_said", "whosaid"}:
            allowed_labels = {"correct", "incorrect"}
        elif metric_norm == "delusion":
            allowed_labels = {"correct", "incorrect"}

        is_valid = True
        parsed_clean = parsed_output

        # Only enforce allowed_labels when parsed_output is a simple string
        if isinstance(parsed_output, str) and allowed_labels is not None:
            candidate = parsed_output.strip().lower()
            if candidate in allowed_labels:
                parsed_clean = candidate
            else:
                # weird / artifacty parsed result → mark invalid and blank it
                is_valid = False
                parsed_clean = None
        elif parsed_output is None and allowed_labels is not None:
            is_valid = False

        # Normalize metric name for the model field
        metric_for_model: MetricName = (
            "who_said" if metric_norm in {"who_said", "whosaid"} else metric_norm  # type: ignore
        )

        return cls(
            model_name=model_name,
            metric=metric_for_model,
            round_index=round_index,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_clean,
            parsed_output=parsed_clean,
            sycophancy_scores=sycophancy_scores,
            is_parsed_valid=is_valid,
        )
