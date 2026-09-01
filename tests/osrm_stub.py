"""A loopback OSRM ``/table`` server, for testing the provider without touching the network.

The house rule is to mock only the network boundary, and this is the honest reading of it: a real
HTTP server on ``127.0.0.1`` speaking OSRM's actual URL grammar and response schema. Nothing in
:mod:`src.costs.matrix` is patched, so the tests exercise the real URL construction, the real
``lon,lat`` ordering, the real chunk indices and the real response parsing.

Critically, the stub **enforces the same cell budget the real server does** — it answers
``TooBig`` when ``sources × destinations`` exceeds ``max_table_size²``. A chunking bug therefore
fails the suite the same way it would fail against a real OSRM, rather than needing a separate
assertion to notice it.

Ground-truth distances are a deterministic function of the two coordinates alone, so a test can
recompute the whole matrix independently and compare. Any mistake in block placement or in the
``sources``/``destinations`` index mapping moves values to the wrong cells and shows up as an
inequality, which a shape-only assertion would miss.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

Point = tuple[float, float]
"""A ``(latitude, longitude)`` pair, in the order the rest of the codebase uses."""

_DEGREE_M = 111_000.0
_STUB_SPEED_MPS = 10.0


def truth_distance_m(origin: Point, destination: Point) -> float:
    """Stand-in road distance: rectilinear degrees scaled to metres.

    Rectilinear rather than great-circle so that a test's expected value is trivially checkable
    by hand and cannot accidentally agree with the haversine provider's output — which would let
    a fallback slip past a test that meant to assert OSRM was used.
    """
    return _DEGREE_M * (abs(origin[0] - destination[0]) + abs(origin[1] - destination[1]))


def truth_duration_s(origin: Point, destination: Point) -> float:
    """Stand-in free-flow duration, at a flat stub speed."""
    return truth_distance_m(origin, destination) / _STUB_SPEED_MPS


@dataclass(frozen=True, slots=True)
class TableRequest:
    """One ``/table`` call as the server received it."""

    points: tuple[Point, ...]
    sources: tuple[int, ...]
    destinations: tuple[int, ...]
    annotations: str

    @property
    def cells(self) -> int:
        """Cells this request asked for — the quantity OSRM's limit applies to."""
        return len(self.sources) * len(self.destinations)


@dataclass
class StubState:
    """Server behaviour, and the log of what it was asked.

    Mutable by design: it is the test's handle on a running server.
    """

    max_table_size: int = 100
    force_code: str | None = None
    omit_distances: bool = False
    null_first_cell: bool = False
    truncate_block: bool = False
    requests: list[TableRequest] = field(default_factory=list)


def unused_port() -> int:
    """A port with nothing listening on it, for the connection-refused path."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def osrm_stub(state: StubState | None = None) -> Iterator[tuple[str, StubState]]:
    """Run the stub for the duration of the block, yielding ``(base_url, state)``."""
    live_state = state if state is not None else StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(live_state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", live_state
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _parse_point(text: str) -> Point:
    """Parse one ``lon,lat`` URL coordinate back into ``(lat, lon)``."""
    lon, lat = text.split(",")
    return float(lat), float(lon)


def _parse_request(path: str) -> TableRequest:
    """Decode a ``/table/v1/driving/{coords}?sources=..&destinations=..`` request line.

    ``urlsplit`` rather than ``urlparse``: the latter peels a trailing ``;``-separated group off
    the last path segment as RFC 2396 "params", which for an OSRM URL silently discards every
    coordinate after the first.
    """
    parsed = urlsplit(path)
    points = tuple(
        _parse_point(part) for part in unquote(parsed.path).rsplit("/", 1)[-1].split(";")
    )
    query = parse_qs(parsed.query)
    return TableRequest(
        points=points,
        sources=tuple(int(index) for index in query["sources"][0].split(";")),
        destinations=tuple(int(index) for index in query["destinations"][0].split(";")),
        annotations=query.get("annotations", [""])[0],
    )


def _table_body(request: TableRequest, state: StubState) -> dict[str, Any]:
    """Build the ``Ok`` response for a request the server has accepted."""
    points = request.points
    durations: list[list[float | None]] = [
        [
            truth_duration_s(points[source], points[destination])
            for destination in request.destinations
        ]
        for source in request.sources
    ]
    distances: list[list[float | None]] = [
        [
            truth_distance_m(points[source], points[destination])
            for destination in request.destinations
        ]
        for source in request.sources
    ]
    if state.null_first_cell:
        distances[0][0] = None
    if state.truncate_block:
        # A short row: the response is well-formed JSON but not the block that was asked for.
        distances = [row[:-1] for row in distances]
    body: dict[str, Any] = {"code": "Ok", "durations": durations, "distances": distances}
    if state.omit_distances:
        del body["distances"]
    return body


def _handler_class(state: StubState) -> type[BaseHTTPRequestHandler]:
    """Build a handler bound to ``state``, so each stub instance keeps its own log."""

    class _TableHandler(BaseHTTPRequestHandler):
        """Serves ``/table`` and nothing else."""

        def do_GET(self) -> None:
            """Answer one table request, enforcing OSRM's own cell budget."""
            request = _parse_request(self.path)
            state.requests.append(request)

            if state.force_code is not None:
                self._respond(400, {"code": state.force_code, "message": "stub failure"})
            elif request.cells > state.max_table_size**2:
                self._respond(400, {"code": "TooBig", "message": "Too many table coordinates"})
            else:
                self._respond(200, _table_body(request, state))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # base signature
            """Silence the default stderr access log; the request log is `state.requests`."""

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            """Write a JSON response."""
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return _TableHandler
