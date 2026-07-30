"""Test configuration - set environment to test mode before importing app."""

import os

# Set test environment before any app imports
os.environ["APP_ENVIRONMENT"] = "test"
