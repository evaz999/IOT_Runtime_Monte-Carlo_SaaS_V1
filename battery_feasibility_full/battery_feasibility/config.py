from dataclasses import dataclass


@dataclass
class SimulationDefaults:
    """Default knobs for V1 coin cell simulations."""
    monte_carlo_samples: int = 2000
    time_step_seconds: float = 60.0  # 1 min step
    vmin_system_volts: float = 2.0   # brownout threshold (example)


DEFAULTS = SimulationDefaults()
