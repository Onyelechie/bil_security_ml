import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging for the Edge Agent.
    """

    # Convert "INFO" -> logging.INFO etc.
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger.
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
