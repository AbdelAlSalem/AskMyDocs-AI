class ServiceError(Exception):
    """An application failure with a response safe to show to the user."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class DocumentError(ServiceError):
    """A document operation failure."""


class ConversationError(ServiceError):
    """A conversation operation failure."""
