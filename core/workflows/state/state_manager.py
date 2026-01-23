#!/usr/bin/env python3
"""State Manager for Long-Running Processes.

Generic state persistence and recovery for any long-running pipeline.
Provides atomic saves and checkpoint management for fault-tolerant processing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateManager:
    """Manages persistent state for long-running processes with atomic saves.

    Preserves process state across interruptions, allowing workflows to resume
    from their last checkpoint rather than starting over.
    """

    def __init__(self, state_file: str, process_name: str):
        """
        Initialize state manager.

        Args:
            state_file: Path to state file
            process_name: Name of the process (for validation)
        """
        self.state_file = Path(state_file)
        self.process_name = process_name
        self.state: dict[str, Any] = {
            'process_name': process_name,
            'created': datetime.now().isoformat(),
            'last_save': None,
            'checkpoints': {},
            'stats': {},
            'metadata': {}
        }
        self.load()

    def load(self) -> bool:
        """
        Load state from file if it exists.

        Returns:
            True if state was loaded, False if starting fresh
        """
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    saved_state = json.load(f)
                    if not isinstance(saved_state, dict):
                        logger.warning(
                            f"State file {self.state_file} did not contain a valid object; starting fresh"
                        )
                        return False

                # Validate it's for the same process
                if saved_state.get('process_name') == self.process_name:
                    self.state.update(saved_state)
                    logger.info(f"Loaded state from {self.state_file}")
                    return True
                else:
                    # Process name mismatch - backup existing file and start fresh
                    backup_name = self.state_file.with_suffix(
                        f'.{saved_state.get("process_name", "unknown")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak'
                    )
                    self.state_file.rename(backup_name)
                    logger.error(
                        f"State file process name mismatch: expected '{self.process_name}', "
                        f"found '{saved_state.get('process_name')}'. "
                        f"Backed up existing state to {backup_name}"
                    )
                    return False

            except Exception as e:
                logger.warning(f"Could not load state from {self.state_file}: {e}")

        return False

    def save(self) -> bool:
        """
        Save state atomically to prevent corruption.

        Returns:
            True if save successful
        """
        self.state['last_save'] = datetime.now().isoformat()

        # Write to temp file first
        temp_file = self.state_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)

            # Atomic rename
            temp_file.replace(self.state_file)
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    def _ensure_section(self, key: str) -> dict[str, Any]:
        """Ensure a nested section exists as a dictionary."""
        section = self.state.get(key)
        if not isinstance(section, dict):
            section = {}
            self.state[key] = section
        return section

    def set_checkpoint(self, name: str, value: Any):
        """Set a named checkpoint."""
        checkpoints = self._ensure_section('checkpoints')
        checkpoints[name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }

    def get_checkpoint(self, name: str, default: Any = None) -> Any:
        """Get a named checkpoint value."""
        checkpoint_section = self._ensure_section('checkpoints')
        checkpoint = checkpoint_section.get(name, {})
        return checkpoint.get('value', default)

    def update_stats(self, **kwargs):
        """Update statistics."""
        stats = self._ensure_section('stats')
        stats.update(kwargs)

    def increment_stat(self, name: str, amount: int = 1):
        """Increment a statistic counter."""
        stats = self._ensure_section('stats')
        current = stats.get(name, 0)
        stats[name] = current + amount

    def set_metadata(self, key: str, value: Any):
        """Set metadata value."""
        metadata = self._ensure_section('metadata')
        metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        metadata = self._ensure_section('metadata')
        return metadata.get(key, default)

    def clear(self):
        """Clear state file."""
        if self.state_file.exists():
            self.state_file.unlink()

        # Reset to initial state
        self.state = {
            'process_name': self.process_name,
            'created': datetime.now().isoformat(),
            'last_save': None,
            'checkpoints': {},
            'stats': {},
            'metadata': {}
        }

    def get_progress_summary(self) -> dict[str, Any]:
        """
        Get a summary of current progress.

        Returns:
            Dictionary with progress information
        """
        return {
            'process': self.process_name,
            'created': self.state.get('created'),
            'last_save': self.state.get('last_save'),
            'checkpoints': len(self._ensure_section('checkpoints')),
            'stats': self._ensure_section('stats')
        }


class CheckpointManager:
    """
    Manages checkpoints for iterative processes.

    Useful for tracking which items have been processed in a large dataset,
    allowing resume after interruption.
    """

    def __init__(self, checkpoint_file: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.processed: set[str] = set()
        self.load()

    def load(self) -> bool:
        """Load processed items from checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.processed = {str(item) for item in data}
                    elif isinstance(data, dict):
                        self.processed = {str(item) for item in data.get('processed', [])}
                    else:
                        logger.warning(
                            f"Unexpected checkpoint format in {self.checkpoint_file}; resetting"
                        )
                        self.processed.clear()

                logger.info(f"Loaded {len(self.processed)} checkpoints")
                return True

            except Exception as e:
                logger.warning(f"Could not load checkpoints: {e}")

        return False

    def save(self) -> bool:
        """Save checkpoints atomically."""
        temp_file = self.checkpoint_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w') as f:
                json.dump(sorted(self.processed), f, indent=2)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    def is_processed(self, item: str) -> bool:
        """Check if an item has been processed."""
        return item in self.processed

    def mark_processed(self, item: str):
        """Mark an item as processed."""
        self.processed.add(item)

    def mark_many_processed(self, items: list):
        """Mark multiple items as processed."""
        self.processed.update(items)

    def get_unprocessed(self, items: list) -> list:
        """Filter a list to only unprocessed items."""
        return [item for item in items if item not in self.processed]

    def clear(self):
        """Clear all checkpoints."""
        self.processed.clear()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def get_stats(self) -> dict[str, int]:
        """Get checkpoint statistics."""
        return {
            'total_processed': len(self.processed)
        }
