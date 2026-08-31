"""Exception hierarchy for the optimization pipeline.

Every deliberate failure in this codebase derives from :class:`OptimizationError`. That lets a
caller separate a domain failure — an instance that cannot be served, a provider that will not
answer — from a genuine bug leaking out as ``KeyError`` or ``TypeError``, which should never be
caught.
"""


class OptimizationError(Exception):
    """Root of the hierarchy. Not raised directly."""


class ConfigurationError(OptimizationError):
    """A config object was constructed with values that cannot describe a valid run."""


class InstanceError(OptimizationError):
    """A problem instance is internally inconsistent, or could not be (de)serialised."""


class InfeasibleSolutionError(OptimizationError):
    """A solution violates a hard constraint: capacity, route structure, or duplicate visits.

    Capacity is enforced by construction upstream, so this being raised means a builder is
    broken — not that the instance is hard.
    """
