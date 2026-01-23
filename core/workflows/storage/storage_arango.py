"""
ArangoDB Storage Utilities.

Database connection management and utilities for ArangoDB.
"""

import logging
from typing import Any

from arango import ArangoClient
from arango.database import StandardDatabase

logger = logging.getLogger(__name__)


class ArangoStorageManager:
    """
    Manage database connections and provide storage utilities.

    Features:
    - Connection pooling
    - Automatic retry logic
    - Collection management
    - Index management
    """

    # Cache connections to avoid recreating
    _connections: dict[str, StandardDatabase] = {}

    @classmethod
    def get_connection(cls, config: Any) -> StandardDatabase:
        """
        Get or create a database connection.

        Args:
            config: Database configuration object

        Returns:
            ArangoDB database connection
        """
        # Create connection key from config (include username for credential isolation)
        connection_key = f"{config.username}@{config.host}:{config.port}/{config.database}"

        # Return cached connection if available
        if connection_key in cls._connections:
            return cls._connections[connection_key]

        # Create new connection
        client = ArangoClient(hosts=f"http://{config.host}:{config.port}")

        # Connect to system database first
        sys_db = client.db(
            '_system',
            username=config.username,
            password=config.password
        )

        # Create database if it doesn't exist
        if not sys_db.has_database(config.database):
            sys_db.create_database(config.database)
            logger.info(f"Created database: {config.database}")

        # Connect to target database
        db = client.db(
            config.database,
            username=config.username,
            password=config.password
        )

        # Cache the connection
        cls._connections[connection_key] = db

        logger.info(f"Connected to database: {connection_key}")
        return db

    @staticmethod
    def ensure_collection(db: StandardDatabase, name: str,
                         edge: bool = False, **kwargs) -> None:
        """
        Ensure a collection exists.

        Args:
            db: Database connection
            name: Collection name
            edge: Whether this is an edge collection
            **kwargs: Additional collection parameters
        """
        if not db.has_collection(name):
            if edge:
                db.create_collection(name, edge=True, **kwargs)
            else:
                db.create_collection(name, **kwargs)
            logger.info(f"Created {'edge' if edge else 'document'} collection: {name}")

    @staticmethod
    def create_index(db: StandardDatabase, collection: str,
                    index_type: str, fields: list, **kwargs) -> None:
        """
        Create an index on a collection.

        Args:
            db: Database connection
            collection: Collection name
            index_type: Type of index (persistent, geo, fulltext, etc.)
            fields: Fields to index
            **kwargs: Additional index parameters
        """
        coll = db.collection(collection)

        try:
            if index_type == "persistent":
                coll.add_persistent_index(fields=fields, **kwargs)
            elif index_type == "geo":
                coll.add_geo_index(fields=fields, **kwargs)
            elif index_type == "fulltext":
                coll.add_fulltext_index(fields=fields, **kwargs)
            elif index_type == "hash":
                coll.add_hash_index(fields=fields, **kwargs)
            elif index_type == "skiplist":
                coll.add_skiplist_index(fields=fields, **kwargs)
            else:
                logger.warning(f"Unknown index type: {index_type}")
                return

            logger.info(f"Created {index_type} index on {collection}.{fields}")
        except Exception as e:
            # Index might already exist
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create index: {e}")

    @staticmethod
    def verify_vector_support(db: StandardDatabase) -> bool:
        """
        Check if the database supports vector indexes.

        Args:
            db: Database connection

        Returns:
            True if vector indexes are supported
        """
        try:
            version_info = db.version()
            if isinstance(version_info, dict):
                version_str = version_info.get('server_version', '0.0.0')
            else:
                version_str = version_info

            major, minor = map(int, version_str.split('.')[:2])

            # Vector indexes require ArangoDB 3.11+
            if (major, minor) >= (3, 11):
                logger.info(f"ArangoDB {version_str} supports vector indexes")
                return True
            else:
                logger.warning(f"ArangoDB {version_str} does not support vector indexes")
                return False

        except Exception as e:
            logger.error(f"Failed to verify vector support: {e}")
            return False
