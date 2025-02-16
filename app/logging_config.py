import logging
from flask import Flask

def setup_logging():
    """
    Set up logging configuration for the Flask app.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Logging is set up.")