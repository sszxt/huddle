"""What a node remembers about itself between runs.

Kept apart from the config file on purpose: config is what a person decided,
state is what the node worked out or was last asked to do. Three things:

- ``node_id``, a random identity made once. Hostnames are not identities —
  two machines cloned from one image, or two fresh Ubuntu installs, can share
  one — so peers recognise each other by this instead.
- the model to bring back after a reboot, set when someone loads a model on
  this node and cleared when it is stopped or handed over to another node.
- the peers that model last ran with, so a reboot waits for them rather than
  planning around a peer that is merely slower to boot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ValidationError

from huddle.config import PeerConfig

log = logging.getLogger("huddle.state")


class NodeState(BaseModel):
    node_id: str
    resume_model: str | None = None
    # When resume_model was set. If two nodes both think they should resume,
    # the more recent decision wins.
    resume_since: float | None = None
    last_members: list[PeerConfig] = []


class StateStore:
    """``state.json`` in the node's state directory, written atomically."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / "state.json"
        self._state: NodeState | None = None

    def load(self) -> NodeState:
        if self._state is not None:
            return self._state
        try:
            self._state = NodeState.model_validate(json.loads(self.path.read_text()))
        except FileNotFoundError:
            self._state = NodeState(node_id=uuid.uuid4().hex)
            self._save(self._state)
        except (OSError, ValueError, ValidationError) as exc:
            # A corrupt file must not stop a node from starting. It does cost
            # the identity, which peers will see as a new node; say so.
            log.warning(
                "state: %s is unreadable (%s); starting with a new identity", self.path, exc
            )
            self._state = NodeState(node_id=uuid.uuid4().hex)
            self._save(self._state)
        return self._state

    def set_resume(self, model: str, members: list[PeerConfig]) -> None:
        state = self.load()
        state.resume_model, state.resume_since = model, time.time()
        state.last_members = list(members)
        self._save(state)

    def clear_resume(self) -> None:
        state = self.load()
        if state.resume_model is None and not state.last_members:
            return
        state.resume_model, state.resume_since, state.last_members = None, None, []
        self._save(state)

    def _save(self, state: NodeState) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".state-")
            with os.fdopen(fd, "w") as handle:
                handle.write(state.model_dump_json(indent=2) + "\n")
            os.replace(temp, self.path)
        except OSError as exc:
            # Losing the resume flag is an inconvenience; crashing is not.
            log.warning("state: could not write %s: %s", self.path, exc)
            with contextlib.suppress(OSError, UnboundLocalError):
                os.unlink(temp)


__all__ = ["NodeState", "StateStore"]
