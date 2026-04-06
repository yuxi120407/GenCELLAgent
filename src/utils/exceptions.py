"""Custom exceptions for the GenCellAgent."""


class AgentPauseSignal(BaseException):
    """
    Signal that the agent should pause for human approval.

    Extends BaseException (not Exception) so it is not caught by
    generic Exception handlers, allowing it to propagate to the UI layer.
    """
    pass
