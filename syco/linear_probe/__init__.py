"""Linear assumption probes and causal steering for the sycophancy study.

This package deliberately has its own artifact namespace.  The historical
structured-label tables are never an implicit input: a run begins with raw
label-model completions produced by :mod:`syco.linear_probe.labels`.
"""

from syco.prompts import STRUCTURED_DIMENSIONS

DIMENSIONS = tuple(
    dimension
    for instrument in ("4dims", "supporttypes")
    for dimension in STRUCTURED_DIMENSIONS[instrument]
)


def dimensions_for_instruments(instruments) -> tuple[str, ...]:
    """Return dimensions in registered instrument order, without duplicates."""
    return tuple(dict.fromkeys(
        dimension
        for instrument in instruments
        for dimension in STRUCTURED_DIMENSIONS[instrument]
    ))


__all__ = ["DIMENSIONS", "dimensions_for_instruments"]
