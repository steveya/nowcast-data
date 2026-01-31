from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AggRule = Literal["mean", "last", "sum"]
FeatureRecipeKind = Literal["rate", "index", "level", "flow"]


@dataclass(frozen=True)
class FeatureRecipe:
    kind: FeatureRecipeKind
    agg: AggRule
    qoq: bool = True
    yoy: bool = True
    level: bool = True
    change: Literal["diff", "pct", "logdiff_saar", "logdiff"] = "pct"


# Used when no series-specific recipe is registered.
DEFAULT_RECIPE = FeatureRecipe(kind="level", agg="mean", change="pct")

RECIPE_REGISTRY: dict[str, FeatureRecipe] = {
    "unrate": FeatureRecipe(kind="rate", agg="mean", change="diff"),
    "ir": FeatureRecipe(kind="rate", agg="mean", change="diff"),
    "cpiaucsl": FeatureRecipe(kind="index", agg="mean", change="logdiff_saar"),
    "cpilfesl": FeatureRecipe(kind="index", agg="mean", change="logdiff_saar"),
    "pcepi": FeatureRecipe(kind="index", agg="mean", change="logdiff_saar"),
    "pcepilfe": FeatureRecipe(kind="index", agg="mean", change="logdiff_saar"),
    "payems": FeatureRecipe(kind="level", agg="mean", change="pct"),
    "indpro": FeatureRecipe(kind="index", agg="mean", change="pct"),
    "rsafs": FeatureRecipe(kind="flow", agg="mean", change="pct"),
    "houst": FeatureRecipe(kind="level", agg="mean", change="pct"),
    "permit": FeatureRecipe(kind="level", agg="mean", change="pct"),
}


def get_recipe(series_key: str) -> FeatureRecipe:
    return RECIPE_REGISTRY.get(series_key.strip().lower(), DEFAULT_RECIPE)


def build_agg_spec_from_recipes(
    predictor_pit_keys: list[str], pit_to_canonical: dict[str, str]
) -> dict[str, str]:
    agg_spec: dict[str, str] = {}
    for pit_key in predictor_pit_keys:
        canonical = pit_to_canonical.get(pit_key, pit_key).strip().lower()
        recipe = get_recipe(canonical)
        agg_spec[pit_key] = recipe.agg
    return agg_spec
