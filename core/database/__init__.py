"""
Core Database Module
====================

Shared database utilities for all HADES-Lab processing pipelines.
Provides factory pattern for database connections and optimized clients.
"""

from .database_factory import DatabaseFactory

__all__ = ['DatabaseFactory']
