"""The problem instance: hubs, sources, customers, shipments — and its JSON form.

An ``Instance`` is the immutable ground truth a run is measured against, so it carries the
generation parameters that produced it (``geo``, ``fleet``, ``schedule``) alongside the nodes.
A results table is only defensible if the instance behind it can be reconstructed byte for
byte, and a bare list of coordinates cannot do that.

The flat node space defined here — ``hubs | sources | customers`` — is the single addressing
scheme used by every distance and duration matrix in the codebase. Solvers index matrices with
:data:`src.units.NodeId` values built by the accessors below and never with raw arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from src.config import FleetConfig, GeoConfig, ScheduleConfig
from src.exceptions import InstanceError
from src.units import NodeId, Seconds

FORMAT_VERSION: Final = 1

# Coordinates are generated inside the bounding box and compared back against it; the slack
# absorbs float round-tripping through JSON, nothing more.
_BBOX_TOLERANCE_DEG: Final = 1e-9


def _require(condition: bool, message: str) -> None:
    """Raise :class:`InstanceError` unless ``condition`` holds."""
    if not condition:
        raise InstanceError(message)


@dataclass(frozen=True, slots=True)
class Coordinate:
    """A WGS84 point in decimal degrees."""

    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A delivery window, in seconds from midnight of the operating day.

    Windows are soft: arriving outside one is penalised, never forbidden. The window therefore
    owns the lateness measurement rather than any solver, so every caller agrees on what "late"
    means.
    """

    start_s: Seconds
    end_s: Seconds

    def __post_init__(self) -> None:
        _require(self.start_s >= 0.0, "time window start_s must be non-negative")
        _require(self.start_s < self.end_s, "time window start_s must precede end_s")

    def lateness_s(self, arrival_s: Seconds) -> Seconds:
        """Seconds by which ``arrival_s`` misses the window, or zero if it does not.

        Early arrival is not lateness: the vehicle waits, which the duration model charges for
        separately.
        """
        return Seconds(max(0.0, arrival_s - self.end_s))


@dataclass(frozen=True, slots=True)
class Hub:
    """A consolidation point: the terminus of Stage 1 tours and the depot of Stage 2 tours."""

    hub_id: int
    coord: Coordinate


@dataclass(frozen=True, slots=True)
class Source:
    """A pickup location. Sources have no time window — inbound collection is untimed."""

    source_id: int
    coord: Coordinate


@dataclass(frozen=True, slots=True)
class Customer:
    """A delivery location, with an optional time window. ``None`` means available all day."""

    customer_id: int
    coord: Coordinate
    window: TimeWindow | None


@dataclass(frozen=True, slots=True)
class Shipment:
    """One parcel: picked up at ``source_id`` in Stage 1, delivered to ``customer_id`` in Stage 2.

    Precedence between the two legs is guaranteed structurally by stage ordering, so a shipment
    needs no state and no repair operator ever inspects it.
    """

    shipment_id: int
    source_id: int
    customer_id: int
    size_kg: float


@dataclass(frozen=True, slots=True)
class Instance:
    """A complete, self-describing problem instance.

    Validation is exhaustive at construction: contiguous ids, in-box coordinates, resolvable
    shipment endpoints, and shipments that fit a vehicle. Everything downstream may then treat
    the instance as correct and index into it without defensive checks.
    """

    seed: int
    geo: GeoConfig
    fleet: FleetConfig
    schedule: ScheduleConfig
    hubs: tuple[Hub, ...]
    sources: tuple[Source, ...]
    customers: tuple[Customer, ...]
    shipments: tuple[Shipment, ...]

    def __post_init__(self) -> None:
        self._validate_counts()
        self._validate_ids()
        self._validate_geography()
        self._validate_shipments()

    def _validate_counts(self) -> None:
        _require(len(self.hubs) == self.geo.n_hubs, "hub count disagrees with GeoConfig.n_hubs")
        _require(
            len(self.sources) == self.geo.n_sources,
            "source count disagrees with GeoConfig.n_sources",
        )
        _require(
            len(self.customers) == self.geo.n_customers,
            "customer count disagrees with GeoConfig.n_customers",
        )
        _require(len(self.shipments) > 0, "an instance must contain at least one shipment")

    def _validate_ids(self) -> None:
        _require(
            [hub.hub_id for hub in self.hubs] == list(range(len(self.hubs))),
            "hub ids must be contiguous and ordered from zero",
        )
        _require(
            [source.source_id for source in self.sources] == list(range(len(self.sources))),
            "source ids must be contiguous and ordered from zero",
        )
        _require(
            [customer.customer_id for customer in self.customers]
            == list(range(len(self.customers))),
            "customer ids must be contiguous and ordered from zero",
        )
        _require(
            [s.shipment_id for s in self.shipments] == list(range(len(self.shipments))),
            "shipment ids must be contiguous and ordered from zero",
        )

    def _validate_geography(self) -> None:
        lat_lo = self.geo.lat_min - _BBOX_TOLERANCE_DEG
        lat_hi = self.geo.lat_max + _BBOX_TOLERANCE_DEG
        lon_lo = self.geo.lon_min - _BBOX_TOLERANCE_DEG
        lon_hi = self.geo.lon_max + _BBOX_TOLERANCE_DEG
        for coord in self._all_coordinates():
            _require(
                lat_lo <= coord.lat <= lat_hi and lon_lo <= coord.lon <= lon_hi,
                f"coordinate {coord} lies outside the configured bounding box",
            )

    def _validate_shipments(self) -> None:
        for shipment in self.shipments:
            _require(
                0 <= shipment.source_id < len(self.sources),
                f"shipment {shipment.shipment_id} references an unknown source",
            )
            _require(
                0 <= shipment.customer_id < len(self.customers),
                f"shipment {shipment.shipment_id} references an unknown customer",
            )
            _require(
                0.0 < shipment.size_kg <= self.fleet.vehicle_capacity_kg,
                f"shipment {shipment.shipment_id} cannot be carried by any vehicle",
            )

    def _all_coordinates(self) -> tuple[Coordinate, ...]:
        """Every coordinate in flat node order: hubs, then sources, then customers.

        Hub, Source and Customer share a ``coord`` field but deliberately have no common base
        class — they are distinct domain nouns, not an inheritance hierarchy — so the three
        groups are concatenated explicitly rather than iterated polymorphically.
        """
        return (
            tuple(hub.coord for hub in self.hubs)
            + tuple(source.coord for source in self.sources)
            + tuple(customer.coord for customer in self.customers)
        )

    @property
    def n_nodes(self) -> int:
        """Side length of the distance and duration matrices."""
        return len(self.hubs) + len(self.sources) + len(self.customers)

    @property
    def n_deliveries(self) -> int:
        """Number of drops a complete solution must make — the denominator of the headline KPI."""
        return len(self.customers)

    def hub_node(self, hub_id: int) -> NodeId:
        """Matrix index of a hub."""
        _require(0 <= hub_id < len(self.hubs), f"unknown hub_id {hub_id}")
        return NodeId(hub_id)

    def source_node(self, source_id: int) -> NodeId:
        """Matrix index of a source."""
        _require(0 <= source_id < len(self.sources), f"unknown source_id {source_id}")
        return NodeId(len(self.hubs) + source_id)

    def customer_node(self, customer_id: int) -> NodeId:
        """Matrix index of a customer."""
        _require(0 <= customer_id < len(self.customers), f"unknown customer_id {customer_id}")
        return NodeId(len(self.hubs) + len(self.sources) + customer_id)

    def is_source_node(self, node: NodeId) -> bool:
        """Whether ``node`` addresses a source, and so is a legal Stage 1 stop."""
        return len(self.hubs) <= node < len(self.hubs) + len(self.sources)

    def is_customer_node(self, node: NodeId) -> bool:
        """Whether ``node`` addresses a customer, and so counts as a drop."""
        return len(self.hubs) + len(self.sources) <= node < self.n_nodes

    def customer_at(self, node: NodeId) -> Customer:
        """The customer addressed by ``node``.

        Scoring resolves windows through here rather than by re-deriving the offset, so an
        off-by-one shows up as an exception instead of as a plausible cost.
        """
        _require(self.is_customer_node(node), f"node {node} is not a customer node")
        return self.customers[node - len(self.hubs) - len(self.sources)]

    def coordinates(self) -> npt.NDArray[np.float64]:
        """All coordinates as an ``(n_nodes, 2)`` array of ``[lat, lon]``, in flat node order.

        Materialise once and pass the array around; the matrix builder is vectorised over it.
        """
        return np.array([(c.lat, c.lon) for c in self._all_coordinates()], dtype=np.float64)

    def coordinate_digest(self) -> str:
        """Stable digest of the node geometry, for keying the cached cost matrix.

        Two instances with the same seed but different counts or box must not collide on a
        cache entry, and this is the part of the key that catches it.
        """
        return hashlib.sha256(self.coordinates().tobytes()).hexdigest()[:16]

    def to_json(self, indent: int | None = None) -> str:
        """Serialise to JSON, including the configs that shaped the data."""
        payload = {"format_version": FORMAT_VERSION, **asdict(self)}
        return json.dumps(payload, indent=indent)

    def write_json(self, path: Path) -> None:
        """Write the instance to ``path``, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> Instance:
        """Rebuild an instance from :meth:`to_json` output, re-running full validation."""
        # JSON is untyped by nature; every field is narrowed by the constructors below and
        # then re-validated by __post_init__.
        payload: dict[str, Any] = json.loads(text)
        version = payload.get("format_version")
        _require(
            version == FORMAT_VERSION,
            f"unsupported instance format_version {version!r}, expected {FORMAT_VERSION}",
        )
        return cls(
            seed=int(payload["seed"]),
            geo=GeoConfig(**payload["geo"]),
            fleet=FleetConfig(**payload["fleet"]),
            schedule=ScheduleConfig(**payload["schedule"]),
            hubs=tuple(Hub(h["hub_id"], _coordinate(h["coord"])) for h in payload["hubs"]),
            sources=tuple(
                Source(s["source_id"], _coordinate(s["coord"])) for s in payload["sources"]
            ),
            customers=tuple(
                Customer(c["customer_id"], _coordinate(c["coord"]), _window(c["window"]))
                for c in payload["customers"]
            ),
            shipments=tuple(Shipment(**s) for s in payload["shipments"]),
        )

    @classmethod
    def read_json(cls, path: Path) -> Instance:
        """Read an instance from ``path``."""
        return cls.from_json(path.read_text(encoding="utf-8"))


# dict[str, Any] below: a decoded JSON object is genuinely untyped. Each field is narrowed here
# and the resulting Instance is re-validated by __post_init__.
def _coordinate(payload: dict[str, Any]) -> Coordinate:
    """Narrow a decoded JSON object to a :class:`Coordinate`."""
    return Coordinate(lat=float(payload["lat"]), lon=float(payload["lon"]))


def _window(payload: dict[str, Any] | None) -> TimeWindow | None:
    """Narrow a decoded JSON object to a :class:`TimeWindow`, preserving all-day as ``None``."""
    if payload is None:
        return None
    return TimeWindow(
        start_s=Seconds(float(payload["start_s"])),
        end_s=Seconds(float(payload["end_s"])),
    )
