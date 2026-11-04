from . import hybrid, react, workflow

ARMS = {
    "workflow": workflow.run,
    "react": react.run,
    "hybrid": hybrid.run,
}

__all__ = ["ARMS", "workflow", "react", "hybrid"]
