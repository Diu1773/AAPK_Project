"""
Project state management for workflow tracking and persistence
Workflow state save/load system
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ProjectState:
    """
    Manages workflow state for step-by-step processing
    Tracks completion status and saves/loads progress
    """

    def __init__(self, project_dir: Path):
        """
        Initialize project state

        Args:
            project_dir: Directory containing project data and state file
        """
        self.project_dir = Path(project_dir)
        self.state_file = self.project_dir / "project_state.json"

        # Workflow steps
        self.steps = [
            "file_selection",
            "crop",
            "sky_preview",
            "detection",
            "wcs_plate_solve",
            "ref_build",
            "star_id_match",
            "target_selection",
            "forced_photometry",
            "aperture_overlay",
            "light_curve",
            "detrend_merge",
        ]

        # Initialize state
        self.state = {
            "project_name": "Untitled Project",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "current_step": 0,  # Index of current step
            "completed_steps": [],  # List of completed step indices
            "step_data": {},  # Step-specific data storage
        }

        # Load existing state if available
        if self.state_file.exists():
            self.load()

    def is_step_completed(self, step_index: int) -> bool:
        """Check if a step is completed"""
        return step_index in self.state["completed_steps"]

    def is_step_accessible(self, step_index: int) -> bool:
        """
        Check if a step can be accessed
        A step is accessible if:
        - It's step 0 (first step always accessible)
        - The previous step is completed
        """
        if step_index == 0:
            return True
        return self.is_step_completed(step_index - 1)

    def mark_step_completed(self, step_index: int):
        """Mark a step as completed"""
        if step_index not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_index)
            self.state["completed_steps"].sort()
        self.state["last_modified"] = datetime.now().isoformat()
        self.save()

    def mark_step_incomplete(self, step_index: int):
        """Mark a step as incomplete"""
        if step_index in self.state["completed_steps"]:
            self.state["completed_steps"].remove(step_index)
        self.state["last_modified"] = datetime.now().isoformat()
        self.save()

    def set_current_step(self, step_index: int):
        """Set the current active step"""
        if 0 <= step_index < len(self.steps):
            self.state["current_step"] = step_index
            self.state["last_modified"] = datetime.now().isoformat()
            self.save()

    def get_current_step(self) -> int:
        """Get current step index"""
        return self.state["current_step"]

    def get_next_incomplete_step(self) -> Optional[int]:
        """Get the index of the next incomplete step"""
        for i in range(len(self.steps)):
            if not self.is_step_completed(i):
                return i
        return None

    def can_proceed_to_next(self) -> bool:
        """Check if can proceed to next step"""
        current = self.state["current_step"]
        return self.is_step_completed(current) and current < len(self.steps) - 1

    def can_go_to_previous(self) -> bool:
        """Check if can go to previous step"""
        return self.state["current_step"] > 0

    def store_step_data(self, step_name: str, data: Dict[str, Any]):
        """
        Store step-specific data

        Args:
            step_name: Name of the step
            data: Dictionary of data to store
        """
        if step_name not in self.state["step_data"]:
            self.state["step_data"][step_name] = {}

        self.state["step_data"][step_name].update(data)
        self.state["last_modified"] = datetime.now().isoformat()
        self.save()

    def get_step_data(self, step_name: str) -> Dict[str, Any]:
        """
        Retrieve step-specific data

        Args:
            step_name: Name of the step

        Returns:
            Dictionary of stored data
        """
        return self.state["step_data"].get(step_name, {})

    def save(self):
        """Save state to JSON file"""
        self.project_dir.mkdir(parents=True, exist_ok=True)

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def load(self):
        """Load state from JSON file"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
                self.state.update(loaded_state)
            self._normalize_state()
        except Exception as e:
            print(f"Warning: Could not load project state: {e}")

    def _normalize_state(self) -> None:
        """Clamp loaded state to the current workflow definition."""
        max_index = len(self.steps) - 1
        completed = [
            i for i in self.state.get("completed_steps", [])
            if isinstance(i, int) and 0 <= i <= max_index
        ]
        self.state["completed_steps"] = sorted(set(completed))

        current = self.state.get("current_step", 0)
        if not isinstance(current, int) or current < 0:
            current = 0
        if current > max_index:
            current = max_index
        self.state["current_step"] = current

    def reset(self):
        """Reset all progress (but keep project info)"""
        project_name = self.state["project_name"]
        created = self.state["created"]

        self.state = {
            "project_name": project_name,
            "created": created,
            "last_modified": datetime.now().isoformat(),
            "current_step": 0,
            "completed_steps": [],
            "step_data": {},
        }
        self.save()

    def export_summary(self) -> str:
        """Export human-readable summary"""
        lines = []
        lines.append(f"Project: {self.state['project_name']}")
        lines.append(f"Created: {self.state['created']}")
        lines.append(f"Last Modified: {self.state['last_modified']}")
        lines.append(f"\nProgress: {len(self.state['completed_steps'])}/{len(self.steps)} steps completed")
        lines.append("\nSteps:")

        for i, step_name in enumerate(self.steps):
            status = "✓" if i in self.state["completed_steps"] else "○"
            current = " (current)" if i == self.state["current_step"] else ""
            lines.append(f"  {status} {i+1}. {step_name.replace('_', ' ').title()}{current}")

        return "\n".join(lines)
