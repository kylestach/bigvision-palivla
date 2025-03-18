import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class State:
    """Represents the state of any entity (gripper or object) at a single timestep."""

    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        if self.width is not None and self.height is not None:
            return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class TrajectoryData:
    """Contains the full trajectory data including gripper, objects, and actions."""

    gripper: List[State]  # List of gripper states over time
    objects: Dict[str, List[State]]  # Object name -> list of states
    action: List[str]  # List of action tokens

    @property
    def num_timesteps(self) -> int:
        """Return the number of timesteps in the trajectory."""
        return len(self.gripper)

    def get_gripper_array(self) -> np.ndarray:
        """Get gripper states as a numpy array."""
        return np.array([state.to_array() for state in self.gripper], dtype=np.float32)

    def get_object_array(self, object_name: str) -> np.ndarray:
        """Get object states as a numpy array."""
        if object_name not in self.objects:
            raise KeyError(f"Object {object_name} not found in trajectory")
        return np.array(
            [state.to_array() for state in self.objects[object_name]], dtype=np.float32
        )


def parse_cot_string(cot_string: str) -> TrajectoryData:
    """
    Parse a chain-of-thought string into a structured format with trajectories.

    Args:
        cot_string: String with trajectory points separated by semicolons, e.g.:
        "gripper<loc0264><loc0360>;silver tray<loc0788><loc0400><loc1020><loc0816>;gripper<loc0300><loc0428>..."

    Returns:
        TrajectoryData containing gripper positions, object states, and action tokens
    """
    # Split the string by semicolons to separate different components
    components = cot_string.split(";")

    gripper_states = []
    objects: Dict[str, List[State]] = {}
    actions = []

    # Process each component until we hit begin_of_action
    for component in components:
        if "<begin_of_action>" in component:
            # Extract all action tokens and break
            actions = re.findall(r"<act\d+>", component)
            break

        # Parse gripper coordinates
        if component.startswith("gripper"):
            loc_tokens = re.findall(r"<loc(\d{4})>", component)
            if len(loc_tokens) >= 2:
                y = float(loc_tokens[0])
                x = float(loc_tokens[1])
                gripper_states.append(State(x=x, y=y))

        # Parse object coordinates
        else:
            match = re.match(r"([^<]+)((?:<loc\d{4}>)+)", component)
            if match:
                obj_name = match.group(1)
                loc_tokens = re.findall(r"<loc(\d{4})>", match.group(2))

                # Initialize object trajectory if not exists
                if obj_name not in objects:
                    objects[obj_name] = []

                if len(loc_tokens) >= 4:
                    y = float(loc_tokens[0])
                    x = float(loc_tokens[1])
                    height = float(loc_tokens[2]) - float(loc_tokens[0])
                    width = float(loc_tokens[3]) - float(loc_tokens[1])
                    objects[obj_name].append(
                        State(x=x, y=y, width=width, height=height)
                    )

    return TrajectoryData(gripper=gripper_states, objects=objects, action=actions)
