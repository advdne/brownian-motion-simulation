"""
Brownian Motion and the Second Law of Thermodynamics Simulation (2D)
--------------------------------------------------------------------
Architecture:
- Functional Core: Pure state transformations, boundary reflection logic,
  and entropy estimation using immutable dataclasses.
- Imperative Shell: Matplotlib setup, real-time animation loop, and rendering I/O.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ==========================================
# 1. CONFIGURATION & DOMAIN VALUE OBJECTS
#    (Functional Core - Immutable Types)
# ==========================================


@dataclass(frozen=True)
class SimulationConfig:
    room_width: float = 100.0
    room_height: float = 100.0
    num_particles: int = 400
    step_size: float = 1.0          # Decreased from 2.5 to 1.0 for finer, slower steps
    initial_box_size: float = 20.0
    initial_box_x: float = 10.0
    initial_box_y: float = 10.0


@dataclass(frozen=True)
class Particle:
    """Immutable representation of a single particle position."""

    x: float
    y: float


@dataclass(frozen=True)
class SimulationState:
    """Immutable snapshot of the entire simulation state at a given time step."""

    particles: Tuple[Particle, ...]
    step_count: int


# ==========================================
# 2. PURE BUSINESS LOGIC & CORE FUNCTIONS
#    (Functional Core - Pure Transformations)
# ==========================================


def initialize_simulation(
    config: SimulationConfig, seed: int | None = None
) -> SimulationState:
    """
    Initializes particles tightly clustered in a localized area of the room,
    representing low initial entropy (concentrated state).
    """
    rng = np.random.default_rng(seed)

    # Generate particles uniformly within the initial localized box
    x_coords = rng.uniform(
        config.initial_box_x,
        config.initial_box_x + config.initial_box_size,
        config.num_particles,
    )
    y_coords = rng.uniform(
        config.initial_box_y,
        config.initial_box_y + config.initial_box_size,
        config.num_particles,
    )

    particles = tuple(
        Particle(x=float(x), y=float(y)) for x, y in zip(x_coords, y_coords)
    )
    return SimulationState(particles=particles, step_count=0)


def reflect_coordinate(
    coord: float, min_val: float, max_val: float
) -> tuple[float, bool]:
    """
    Pure function to handle elastic wall collisions.
    If a particle hits or crosses a boundary, it reflects back inside.
    """
    if coord < min_val:
        return min_val + (min_val - coord), True
    elif coord > max_val:
        return max_val - (coord - max_val), True
    return coord, False


def update_particle_position(
    particle: Particle, dx: float, dy: float, config: SimulationConfig
) -> Particle:
    """
    Computes a new particle position given a random displacement,
    applying boundary reflection rules.
    """
    new_x = particle.x + dx
    new_y = particle.y + dy

    # Enforce boundaries (elastic collisions with walls)
    new_x, _ = reflect_coordinate(new_x, 0.0, config.room_width)
    new_y, _ = reflect_coordinate(new_y, 0.0, config.room_height)

    return Particle(x=new_x, y=new_y)


def step_simulation(
    state: SimulationState, config: SimulationConfig, rng: np.random.Generator
) -> SimulationState:
    """
    Pure transition function: Takes the current simulation state and returns a
    new updated SimulationState after one time step of Brownian motion.
    """
    # Generate random thermal kicks (Gaussian white noise) for all particles simultaneously
    dxs = rng.normal(loc=0.0, scale=config.step_size, size=len(state.particles))
    dys = rng.normal(loc=0.0, scale=config.step_size, size=len(state.particles))

    new_particles = tuple(
        update_particle_position(p, dx, dy, config)
        for p, (dx, dy) in zip(state.particles, zip(dxs, dys))
    )

    return SimulationState(particles=new_particles, step_count=state.step_count + 1)


def calculate_spatial_variance(state: SimulationState) -> float:
    """
    Calculates the spatial variance of particles as a proxy for entropy/spread.
    As particles spread out to fill the room, variance increases and stabilizes,
    illustrating the Second Law of Thermodynamics.
    """
    xs = [p.x for p in state.particles]
    ys = [p.y for p in state.particles]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
    var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)

    return float(var_x + var_y)


# ==========================================
# 3. IMPERATIVE SHELL (Visualization & I/O)
# ==========================================


def run_simulation() -> None:
    """
    Sets up the Matplotlib imperative animation shell, managing the render loop
    and stepping through the functional core state updates.
    """
    config = SimulationConfig()
    rng = np.random.default_rng(seed=42)  # Reproducible random number stream

    # Initialize state
    state = initialize_simulation(config, seed=42)

    # Setup Figure and Axis
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, config.room_width)
    ax.set_ylim(0, config.room_height)
    ax.set_aspect("equal")

    # Extract initial coordinates for scatter plot
    x_data = [p.x for p in state.particles]
    y_data = [p.y for p in state.particles]

    scat = ax.scatter(
        x_data, y_data, color="royalblue", alpha=0.7, s=15, edgecolors="none"
    )

    # Draw initial concentration boundary outline for visual context
    rect = plt.Rectangle(
        (config.initial_box_x, config.initial_box_y),
        config.initial_box_size,
        config.initial_box_size,
        linewidth=1.5,
        edgecolor="crimson",
        facecolor="none",
        linestyle="--",
        label="Initial Concentration Area",
    )
    ax.add_patch(rect)

    ax.legend(loc="upper right")
    title_text = ax.set_title(
        f"Step: {state.step_count} | Spatial Variance (Entropy Proxy): {calculate_spatial_variance(state):.1f}",
        fontsize=11,
    )
    ax.set_xlabel("X Position (Room Width)")
    ax.set_ylabel("Y Position (Room Height)")

    # Define the animation update closure
    def update(frame_idx: int) -> tuple:
        nonlocal state
        # Invoke pure core state transition
        state = step_simulation(state, config, rng)

        # Update plot data
        updated_x = [p.x for p in state.particles]
        updated_y = [p.y for p in state.particles]

        # Matplotlib expects a 2D array of shape (N, 2) for scatter offsets
        scat.set_offsets(np.column_stack([updated_x, updated_y]))

        # Update title info
        current_variance = calculate_spatial_variance(state)
        title_text.set_text(
            f"Step: {state.step_count} | Spatial Variance (Entropy Proxy): {current_variance:.1f}"
        )
        return scat, title_text

    # Run the animation
    anim = animation.FuncAnimation(
        # fig, update, frames=300, interval=30, blit=True, repeat=False
        fig,
        update,
        frames=1000,  # run longer
        interval=70,  # higher interval means a longer pause between frames, slowing down the visual playback.
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
