"""Safe evaluation of user-defined empirical formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class FormulaError(ValueError):
    """Error in a formula expression or constant definition."""


@dataclass
class FormulaEngine:
    """Evaluates a Python expression in a restricted namespace."""

    expression: str = "2.5 * H**0.67"
    constants: dict[str, float] = field(default_factory=dict)

    def set_constants_from_text(self, text: str) -> None:
        """Parses constants in 'a=1.2, b=3' or one-per-line format."""
        constants: dict[str, float] = {}
        normalized = text.replace("\n", ",")
        for part in normalized.split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise FormulaError(f"Constant is not in name=value format: {item}")
            name, value = [chunk.strip() for chunk in item.split("=", 1)]
            if not name.isidentifier():
                raise FormulaError(f"Invalid constant name: {name}")
            try:
                constants[name] = float(value)
            except ValueError as exc:
                raise FormulaError(f"Constant {name} is not a valid number.") from exc
        self.constants = constants

    def evaluate(self, height_cm: float) -> float:
        """Evaluates the formula for flame height H in centimetres."""
        if height_cm <= 0:
            return float("nan")

        namespace = {
            "H": float(height_cm),
            "math": math,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            **self.constants,
        }
        try:
            result = eval(self.expression, {"__builtins__": {}}, namespace)
        except Exception as exc:  # noqa: BLE001 — GUI needs a readable error.
            raise FormulaError(f"Formula evaluation failed: {exc}") from exc

        try:
            return float(result)
        except (TypeError, ValueError) as exc:
            raise FormulaError("Formula did not return a numeric value.") from exc
