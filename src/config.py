"""Immutable configuration for every stage of the pipeline.

Every tunable in this repository lives here. Nothing reads configuration at import time and
nothing mutates it: a config object is constructed once at an entry point and passed down by
value. That is what makes multi-seed runs comparable and per-hub `multiprocessing` solves
reproducible — a worker cannot be handed a config that a sibling has since changed.

Validation happens in ``__post_init__`` rather than at the point of use, so an impossible run
fails at the entry point with a message naming the offending field instead of surfacing 40
seconds later as a division by zero inside a fitness evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.exceptions import ConfigurationError

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60.0
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
METRES_PER_KM = 1000.0

GRAMS_PER_KG = 1000
"""Scale for handing masses to OR-Tools, whose capacity dimensions are integer-valued.

The default 37.5 kg shipment is not a whole number of kilograms, so rounding demands to integer
kilograms would drift by up to half a kilo per stop and could let a 20-stop tour appear to fit
inside a vehicle it does not. Grams are exact for every shipment size this instance generates.
"""

COST_SCALE_MILLI_INR = 1000
"""Scale for handing arc costs to OR-Tools, whose objective is integer-valued.

One milli-rupee is about 11 cm of driving at ₹9/km — two orders of magnitude below the resolution
of the road network the distances come from, so the rounding cannot change which arc is cheaper.
"""

EARTH_RADIUS_M = 6_371_008.8
"""IUGG mean Earth radius, used by the haversine fallback.

The 0.3% spread between the equatorial and polar radii is an order of magnitude below the error
already introduced by ``circuity_factor``, so a spherical model is the right level of care here.
"""

MIN_TOURNAMENT_K = 2
"""A tournament needs at least two contestants to select between."""

MIN_POPULATION_SIZE = MIN_TOURNAMENT_K
"""A GA population below the smallest tournament cannot support selection at all."""

MIN_ROUTE_NODES = 3
"""Hub, at least one stop, hub. Anything shorter is not a tour."""

CAPACITY_TOLERANCE_KG = 1e-6
"""Slack when comparing a summed float load against capacity.

Loads are accumulated in floating point, so an exactly-full vehicle can land a fraction of a
microgram over. Without this, a legal route built from 20 × 37.5 kg would sometimes be rejected.
"""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigurationError` unless ``condition`` holds.

    Keeps validators to one readable line each, which matters because a config with fifteen
    fields otherwise dwarfs the type it validates.
    """
    if not condition:
        raise ConfigurationError(message)


@dataclass(frozen=True, slots=True)
class GeoConfig:
    """Geography of the synthetic instance: bounding box, node counts, density structure.

    The box covers Mumbai and Navi Mumbai. Sources and customers are not drawn uniformly:
    a configurable share is drawn from a handful of Gaussian density clusters, because uniform
    scatter produces an unrealistically even workload in which consolidation gains nothing and
    the optimizer looks better than it is.
    """

    lat_min: float = 18.90
    lat_max: float = 19.30
    lon_min: float = 72.77
    lon_max: float = 73.10

    n_hubs: int = 16
    n_sources: int = 300
    n_customers: int = 800

    n_density_clusters: int = 4
    clustered_fraction: float = 0.55
    cluster_sigma_deg: float = 0.022

    hub_candidate_pool: int = 4000
    hub_kmeans_iterations: int = 60

    def __post_init__(self) -> None:
        _require(self.lat_min < self.lat_max, "lat_min must be below lat_max")
        _require(self.lon_min < self.lon_max, "lon_min must be below lon_max")
        _require(self.n_hubs > 0, "n_hubs must be positive")
        _require(self.n_sources > 0, "n_sources must be positive")
        _require(self.n_customers > 0, "n_customers must be positive")
        _require(self.n_density_clusters > 0, "n_density_clusters must be positive")
        _require(0.0 <= self.clustered_fraction <= 1.0, "clustered_fraction must be in [0, 1]")
        _require(self.cluster_sigma_deg > 0.0, "cluster_sigma_deg must be positive")
        _require(
            self.hub_candidate_pool >= self.n_hubs,
            "hub_candidate_pool must be at least n_hubs for k-means to place every hub",
        )
        _require(self.hub_kmeans_iterations > 0, "hub_kmeans_iterations must be positive")

    @property
    def n_nodes(self) -> int:
        """Size of the flat node space, and therefore the side length of the cost matrices."""
        return self.n_hubs + self.n_sources + self.n_customers


@dataclass(frozen=True, slots=True)
class FleetConfig:
    """Vehicle and shipment sizing.

    Capacity and shipment size are configuration, never literals, because a mixed fleet and
    mixed shipment sizes are named roadmap extensions. Defaults describe a Tata Ace class SCV:
    750 kg payload, and a 37.5 kg parcel giving exactly 20 shipments per vehicle.
    """

    vehicle_capacity_kg: float = 750.0
    shipment_size_kg: float = 37.5
    vehicle_slack_factor: float = 1.15
    service_time_per_stop_s: float = 300.0

    def __post_init__(self) -> None:
        _require(self.vehicle_capacity_kg > 0.0, "vehicle_capacity_kg must be positive")
        _require(self.shipment_size_kg > 0.0, "shipment_size_kg must be positive")
        _require(
            self.shipment_size_kg <= self.vehicle_capacity_kg,
            "shipment_size_kg exceeds vehicle_capacity_kg: no vehicle could carry one shipment",
        )
        _require(self.vehicle_slack_factor >= 1.0, "vehicle_slack_factor must be at least 1.0")
        _require(self.service_time_per_stop_s >= 0.0, "service_time_per_stop_s must be >= 0")

    @property
    def shipments_per_vehicle(self) -> int:
        """Hard cap on stops per tour when every shipment is the default size."""
        return int(self.vehicle_capacity_kg // self.shipment_size_kg)


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost model in INR.

    ``driver_per_hour`` is time-denominated on purpose: it is the coupling that makes the
    traffic model change the answer rather than decorate it. A distance-only objective would
    be indifferent to when a vehicle drives.
    """

    variable_per_km: float = 9.0
    driver_per_hour: float = 95.0
    fixed_per_vehicle: float = 1000.0
    tw_penalty_per_hour: float = 250.0

    def __post_init__(self) -> None:
        # Zero is permitted so an ablation can switch a component off; negatives are nonsense.
        _require(self.variable_per_km >= 0.0, "variable_per_km must be >= 0")
        _require(self.driver_per_hour >= 0.0, "driver_per_hour must be >= 0")
        _require(self.fixed_per_vehicle >= 0.0, "fixed_per_vehicle must be >= 0")
        _require(self.tw_penalty_per_hour >= 0.0, "tw_penalty_per_hour must be >= 0")


@dataclass(frozen=True, slots=True)
class TrafficBand:
    """One time-of-day travel-time multiplier, over the half-open hour range ``[start, end)``.

    A band may wrap midnight (``start_hour > end_hour``), which is how the overnight band is
    expressed.
    """

    start_hour: int
    end_hour: int
    multiplier: float

    def __post_init__(self) -> None:
        _require(0 <= self.start_hour < HOURS_PER_DAY, "start_hour must be in [0, 24)")
        _require(0 < self.end_hour <= HOURS_PER_DAY, "end_hour must be in (0, 24]")
        _require(self.start_hour != self.end_hour, "a band must span at least one hour")
        _require(self.multiplier > 0.0, "multiplier must be positive")

    @property
    def hours(self) -> tuple[int, ...]:
        """The clock hours this band covers, unrolled and midnight-wrap aware."""
        if self.start_hour < self.end_hour:
            return tuple(range(self.start_hour, self.end_hour))
        return tuple(range(self.start_hour, HOURS_PER_DAY)) + tuple(range(self.end_hour))


DEFAULT_TRAFFIC_BANDS: tuple[TrafficBand, ...] = (
    TrafficBand(start_hour=8, end_hour=11, multiplier=1.6),
    TrafficBand(start_hour=11, end_hour=17, multiplier=1.2),
    TrafficBand(start_hour=17, end_hour=21, multiplier=1.8),
    TrafficBand(start_hour=21, end_hour=8, multiplier=1.0),
)


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    """The time-of-day multiplier schedule.

    This type holds *data only*. Applying a multiplier cumulatively along a route — including
    blending across a band boundary crossed mid-leg — belongs to the cost layer, which is its
    single owner. Bands must partition the day exactly, so no clock hour is ambiguous or
    uncovered.
    """

    bands: tuple[TrafficBand, ...] = DEFAULT_TRAFFIC_BANDS

    def __post_init__(self) -> None:
        _require(len(self.bands) > 0, "at least one traffic band is required")
        covered = [hour for band in self.bands for hour in band.hours]
        _require(
            sorted(covered) == list(range(HOURS_PER_DAY)),
            "traffic bands must cover each of the 24 clock hours exactly once",
        )

    def hourly_multipliers(self) -> tuple[float, ...]:
        """Expand the bands into a 24-entry lookup indexed by clock hour.

        Materialise this once and reuse it. Calling it per leg of a route would put a Python
        loop over the bands inside the innermost loop of the GA.
        """
        table = [0.0] * HOURS_PER_DAY
        for band in self.bands:
            for hour in band.hours:
                table[hour] = band.multiplier
        return tuple(table)


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """The operating day, and the shape of customer delivery time windows.

    Sources carry no windows — inbound pickup is unconstrained in time — so everything here
    describes final-mile delivery only.
    """

    day_start_hour: float = 8.0
    day_end_hour: float = 20.0
    dispatch_hour: float = 8.0

    min_window_hours: float = 2.0
    max_window_hours: float = 4.0
    all_day_fraction: float = 0.25
    window_granularity_minutes: int = 15

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.day_start_hour < self.day_end_hour <= float(HOURS_PER_DAY),
            "operating day must satisfy 0 <= day_start_hour < day_end_hour <= 24",
        )
        _require(
            self.day_start_hour <= self.dispatch_hour <= self.day_end_hour,
            "dispatch_hour must fall inside the operating day",
        )
        _require(0.0 < self.min_window_hours <= self.max_window_hours, "window bounds are inverted")
        _require(
            self.max_window_hours <= self.day_length_hours,
            "max_window_hours cannot exceed the operating day",
        )
        _require(0.0 <= self.all_day_fraction <= 1.0, "all_day_fraction must be in [0, 1]")
        _require(self.window_granularity_minutes > 0, "window_granularity_minutes must be positive")

    @property
    def day_length_hours(self) -> float:
        """Length of the operating day in hours."""
        return self.day_end_hour - self.day_start_hour


@dataclass(frozen=True, slots=True)
class Stage1Config:
    """Inbound consolidation: how hard to balance the hubs, and how long to search each one.

    ``hub_balance_slack`` is the only knob on the min-cost-flow assignment. It multiplies the even
    share of sources per hub to give the cap the flow may not exceed: 1.0 forces a perfectly even
    split regardless of geography, and a large value degenerates to nearest-hub. The quantity
    capped is deliberately a **source count**, not a mass — one indivisible unit of flow per
    source is what makes splitting a source across two hubs structurally impossible, and a
    kilogram-denominated arc bound would split one the moment a cap bound.

    ``cvrp_solution_limit`` is a reproducibility control, not a quality one. Guided local search
    under a wall-clock limit returns whatever it reached before the clock ran out, so the same
    instance on a busier machine yields a different plan. Setting this to 1 stops the search at
    the first-solution heuristic, which is deterministic; the test suite runs that way. Zero means
    unlimited, and is what a real run uses.

    ``workers`` is the size of the per-hub process pool. Hubs are independent, so this is pure
    speedup; zero means one worker per CPU, resolved when the pool is created rather than at
    import, because a module-level ``os.cpu_count()`` would be a config read at import time.
    """

    hub_balance_slack: float = 1.25
    cvrp_time_limit_s: float = 10.0
    cvrp_solution_limit: int = 0
    workers: int = 0

    def __post_init__(self) -> None:
        _require(
            self.hub_balance_slack >= 1.0,
            "hub_balance_slack must be at least 1.0: a cap below the even share is infeasible",
        )
        _require(self.cvrp_time_limit_s > 0.0, "cvrp_time_limit_s must be positive")
        _require(
            self.cvrp_solution_limit >= 0, "cvrp_solution_limit must be >= 0 (0 means unlimited)"
        )
        _require(self.workers >= 0, "workers must be >= 0 (0 means one per CPU)")


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Genetic algorithm hyperparameters for the Stage 2 final-mile solver.

    ``generations`` is a **budget, not a tuned value.** It was written before the arc pricer
    existed and nothing has since been fitted to it; at 600 it costs about 15 minutes on the
    236-stop hub that sets the makespan. If the ablation or the multi-seed evaluation needs
    headroom this is the lever, and moving it is reported rather than absorbed — a figure quoted
    against one budget is not comparable with one quoted against another.

    ``or_opt_max_segment_stops`` bounds the run :func:`~src.stage2.operators.or_opt_mutation`
    relocates. Three is the conventional or-opt neighbourhood: long enough to move a small cluster
    of drops together, short enough that the move stays local and the tour it lands in is still
    recognisably the parent's.
    """

    population_size: int = 150
    generations: int = 600
    tournament_k: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.20
    elitism_count: int = 3
    local_search_pct: float = 0.10
    stagnation_limit: int = 75
    or_opt_max_segment_stops: int = 3

    def __post_init__(self) -> None:
        _require(
            self.population_size >= MIN_POPULATION_SIZE,
            f"population_size must be at least {MIN_POPULATION_SIZE}",
        )
        _require(self.generations > 0, "generations must be positive")
        _require(
            MIN_TOURNAMENT_K <= self.tournament_k <= self.population_size,
            f"tournament_k must be in [{MIN_TOURNAMENT_K}, population_size]",
        )
        _require(0.0 <= self.crossover_rate <= 1.0, "crossover_rate must be in [0, 1]")
        _require(0.0 <= self.mutation_rate <= 1.0, "mutation_rate must be in [0, 1]")
        _require(
            0 <= self.elitism_count < self.population_size,
            "elitism_count must leave room for at least one child",
        )
        _require(0.0 <= self.local_search_pct <= 1.0, "local_search_pct must be in [0, 1]")
        _require(self.stagnation_limit > 0, "stagnation_limit must be positive")
        _require(self.or_opt_max_segment_stops > 0, "or_opt_max_segment_stops must be positive")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Per-run execution settings: seed, distance provider, cache location.

    ``circuity_factor`` and ``haversine_speed_kmph`` describe the fallback only: great-circle
    distance scaled to a road-network estimate, and the average speed that turns it into a
    free-flow duration. 24 km/h is a plausible all-day Mumbai average *before* the traffic
    multipliers are applied on top, since those are what make the peak hours slow.

    1.30 is measured, not assumed: OSRM road distance over great-circle distance across eight
    Mumbai landmarks runs 1.13–1.40 with a mean of 1.28 (``make providers``). It is calibrated
    against *well-connected real places* on purpose. The same ratio over this instance's own
    nodes is far higher — around 1.9 on the legs a tour drives — but that figure is not
    circuity: roughly a quarter of generated nodes sit more than 500 m from any routable road,
    so OSRM measures between snapped positions while the great-circle measures between the
    originals. The contamination is visible as ratios *below* 1.0, which no real road network
    can produce. Fitting this constant to that number would encode a data-generation artefact
    under a name that claims to describe roads. See the README's limitations.

    The default port is 5001, not OSRM's conventional 5000, because macOS binds 5000 to the
    AirPlay Receiver and a default that fails on a fresh clone is not a default. It matches the
    port ``docker-compose.yml`` publishes; change the two together.

    ``osrm_max_table_size`` mirrors the server's own limit. OSRM rejects a ``/table`` request
    whose ``sources × destinations`` cell count exceeds ``max-table-size²``, so this is the side
    length of the largest square block a single request may ask for — not a coordinate budget for
    the whole matrix. The default matches the public demo server, and the bundled
    ``docker-compose.yml`` pins the local server to the same value so development exercises the
    identical constraint.
    """

    seed: int = 42
    osrm_url: str = "http://127.0.0.1:5001"
    use_osrm: bool = True
    osrm_max_table_size: int = 100
    osrm_timeout_s: float = 30.0
    circuity_factor: float = 1.30
    haversine_speed_kmph: float = 24.0
    cache_dir: Path = Path("data/cache")
    data_dir: Path = Path("data")
    figure_dir: Path = Path("figures")

    def __post_init__(self) -> None:
        _require(self.seed >= 0, "seed must be non-negative")
        _require(bool(self.osrm_url.strip()), "osrm_url must not be blank")
        _require(self.osrm_max_table_size > 0, "osrm_max_table_size must be positive")
        _require(self.osrm_timeout_s > 0.0, "osrm_timeout_s must be positive")
        _require(
            self.circuity_factor >= 1.0,
            "circuity_factor must be >= 1.0: road distance is never shorter than great-circle",
        )
        _require(self.haversine_speed_kmph > 0.0, "haversine_speed_kmph must be positive")


@dataclass(frozen=True, slots=True)
class Config:
    """The complete configuration for one run, assembled at an entry point and passed down."""

    geo: GeoConfig = field(default_factory=GeoConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    ga: GAConfig = field(default_factory=GAConfig)
    run: RunConfig = field(default_factory=RunConfig)
