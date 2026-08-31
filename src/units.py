"""Domain scalar aliases.

Distances, durations, money and node indices all reduce to a bare ``float`` or ``int`` at
runtime, which makes them silently interchangeable at call sites such as
``Route(distance_m=..., duration_s=...)``. ``NewType`` makes the distinction visible to
``mypy --strict`` at zero runtime cost, so a duration handed to a distance parameter is a
type error rather than a wrong number in the results table.
"""

from typing import NewType

Metres = NewType("Metres", float)
Seconds = NewType("Seconds", float)
Rupees = NewType("Rupees", float)

NodeId = NewType("NodeId", int)
"""Index into the single flat node space shared by every distance/duration matrix.

Layout is ``hubs | sources | customers`` — see :meth:`src.data.instance.Instance.hub_node`
and friends, which are the only sanctioned way to construct one.
"""
