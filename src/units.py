"""Domain scalar aliases.

Distances, durations, money and node indices all reduce to a bare ``float`` or ``int`` at
runtime, which makes them silently interchangeable at call sites such as
``Route(distance_m=..., duration_s=...)``. ``NewType`` makes the distinction visible to
``mypy --strict`` at zero runtime cost, so a duration handed to a distance parameter is a
type error rather than a wrong number in the results table.
"""

from typing import NewType

import numpy as np
import numpy.typing as npt

Metres = NewType("Metres", float)
Seconds = NewType("Seconds", float)
Rupees = NewType("Rupees", float)

NodeId = NewType("NodeId", int)
"""Index into the single flat node space shared by every distance/duration matrix.

Layout is ``hubs | sources | customers`` — see :meth:`src.data.instance.Instance.hub_node`
and friends, which are the only sanctioned way to construct one.
"""

Coordinates = npt.NDArray[np.float64]
"""An ``(n, 2)`` array of ``[latitude, longitude]`` in decimal degrees, in flat node order.

Latitude first. OSRM's URL grammar is ``lon,lat``; the swap happens once, inside the provider
that needs it, and nowhere else.
"""

NodeArray = npt.NDArray[np.intp]
"""Node ids as a NumPy index array, for indexing a matrix row in one operation."""

DemandArray = npt.NDArray[np.float64]
"""Mass in kilograms, aligned element-wise with a :data:`NodeArray`."""

DistanceMatrix = npt.NDArray[np.float64]
"""An ``(n, n)`` matrix of road distances in metres, indexed by :data:`NodeId`."""

DurationMatrix = npt.NDArray[np.float64]
"""An ``(n, n)`` matrix of **free-flow** travel times in seconds, indexed by :data:`NodeId`.

Free-flow is the contract: no provider applies a time-of-day multiplier. Traffic is layered on
top by :mod:`src.costs.traffic`, cumulatively along a route, because the multiplier depends on
when a leg is driven and a matrix cannot know that.
"""

MatrixPair = tuple[DistanceMatrix, DurationMatrix]
"""``(distance_metres, duration_seconds)`` — the raw two-array result of a provider query.

These three are plain aliases rather than ``NewType``: a ``NewType`` over ``ndarray`` cannot be
indexed or sliced without casting, which would cost more than the confusion it prevents. They
document intent at signatures; :class:`src.costs.matrix.CostMatrices` is what actually enforces
that the two arrays agree.
"""
