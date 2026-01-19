from sdg_core_lib.post_process.functions.Parameter import Parameter
from sdg_core_lib.post_process.functions.UnspecializedFunction import (
    UnspecializedFunction,
    Priority,
)

import numpy as np


class BurstNoiseAdder(UnspecializedFunction):
    parameters = [
        Parameter("magnitude", "30.0", "float"),
        Parameter("n_bursts", "1", "int"),
        Parameter("burst_duration", "1", "int"),
    ]
    description = "Adds n bursts of noise to the data with duration of burst_duration and value of magnitude"
    is_generative = False
    priority = Priority.LOW

    def __init__(self, parameters: list[Parameter]):
        self.magnitude = None
        self.n_bursts = None
        self.burst_duration = None
        super().__init__(parameters)

    def _check_parameters(self):
        allowed_parameters = [param.name for param in type(self).parameters]
        param_mapping = {
            param.name: param
            for param in self.parameters
            if param.name in allowed_parameters
        }
        for name, param in param_mapping.items():
            setattr(self, name, param.value)
        if self.n_bursts < 1:
            raise ValueError("Number of bursts must be at least 1")
        if self.burst_duration < 1:
            raise ValueError("Burst duration must be at least 1")

    def apply(self, n_rows: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.burst_duration > len(data):
            return data, np.array([])

        if self.n_bursts > len(data) // 2:
            return data, np.array([])

        data_copy = np.copy(data)
        affected_indices = set()

        for i in range(self.n_bursts):
            # Find valid starting positions that don't overlap with existing bursts
            max_start = len(data) - self.burst_duration
            valid_positions = []

            for pos in range(max_start + 1):
                burst_range = range(pos, pos + self.burst_duration)
                # Check if this burst would overlap with any existing affected indices
                if not any(idx in affected_indices for idx in burst_range):
                    valid_positions.append(pos)

            if not valid_positions:
                break  # No more valid positions for non-overlapping bursts

            # Choose a random valid position
            idx = np.random.choice(valid_positions)

            # Add the noise burst
            noise = np.ones_like(data[idx : idx + self.burst_duration]) * self.magnitude
            data_copy[idx : idx + self.burst_duration] += noise

            # Track affected indices
            for burst_idx in range(idx, idx + self.burst_duration):
                affected_indices.add(burst_idx)

        return data_copy, np.array(sorted(affected_indices))
