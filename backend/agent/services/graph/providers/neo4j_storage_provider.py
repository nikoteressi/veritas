"""
Neo4j Storage Provider.

Implements StorageInterface using Neo4j database functionality.
"""

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver, Session

from agent.services.graph.interfaces.storage_interface import StorageInterface
from agent.services.graph.interfaces.config_interface import ConfigInterface
from agent.services.graph.dependency_injection import injectable

logger = logging.getLogger(__name__)


@injectable
class Neo4jStorageProvider(StorageInterface):
    """
    Neo4j-based storage provider.

    Provides database connectivity and transaction management using Neo4j
    with support for connection pooling and automatic retry.
    """

    def __init__(self, config: ConfigInterface):
        """
        Initialize Neo4j storage provider.

        Args:
            config: Configuration interface for accessing settings
        """
        self.config = config
        self._driver: Optional[Driver] = None
        self._logger = logging.getLogger(__name__)

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize Neo4j connection."""
        try:
            storage_config = self.config.get_storage_config()

            uri = storage_config.get('uri', 'bolt://localhost:7687')
            username = storage_config.get('username', 'neo4j')
            password = storage_config.get('password', 'password')
            database = storage_config.get('database', 'neo4j')

            # Connection configuration
            max_connection_lifetime = storage_config.get(
                'max_connection_lifetime', 3600)
            max_connection_pool_size = storage_config.get(
                'max_connection_pool_size', 50)
            connection_acquisition_timeout = storage_config.get(
                'connection_acquisition_timeout', 60)

            self._driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_lifetime=max_connection_lifetime,
                max_connection_pool_size=max_connection_pool_size,
                connection_acquisition_timeout=connection_acquisition_timeout
            )

            self._database = database

            # Test connection
            self._driver.verify_connectivity()

            self._logger.info("Connected to Neo4j at %s", uri)

        except Exception as e:
            self._logger.error("Failed to connect to Neo4j: %s", e)
            raise

    def connect(self) -> bool:
        """
        Establish connection to the storage backend.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self._driver is None:
                self._initialize_connection()

            # Verify connectivity
            self._driver.verify_connectivity()
            return True

        except Exception as e:
            self._logger.error("Failed to connect to Neo4j: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        try:
            if self._driver:
                self._driver.close()
                self._driver = None
                self._logger.info("Disconnected from Neo4j")

        except Exception as e:
            self._logger.error("Error disconnecting from Neo4j: %s", e)

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read query.

        Args:
            query: Cypher query to execute
            parameters: Query parameters

        Returns:
            Query results as list of dictionaries

        Raises:
            RuntimeError: If query execution fails
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")

            with self._driver.session(database=self._database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]

                self._logger.debug(
                    "Executed query, returned %d records", len(records))
                return records

        except Exception as e:
            self._logger.error("Failed to execute query: %s", e)
            raise RuntimeError(f"Query execution failed: {e}")

    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a write query.

        Args:
            query: Cypher query to execute
            parameters: Query parameters

        Returns:
            Query execution summary

        Raises:
            RuntimeError: If query execution fails
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")

            with self._driver.session(database=self._database) as session:
                result = session.run(query, parameters or {})
                summary = result.consume()

                summary_data = {
                    'nodes_created': summary.counters.nodes_created,
                    'nodes_deleted': summary.counters.nodes_deleted,
                    'relationships_created': summary.counters.relationships_created,
                    'relationships_deleted': summary.counters.relationships_deleted,
                    'properties_set': summary.counters.properties_set,
                    'labels_added': summary.counters.labels_added,
                    'labels_removed': summary.counters.labels_removed,
                    'indexes_added': summary.counters.indexes_added,
                    'indexes_removed': summary.counters.indexes_removed,
                    'constraints_added': summary.counters.constraints_added,
                    'constraints_removed': summary.counters.constraints_removed
                }

                self._logger.debug("Executed write query: %s", summary_data)
                return summary_data

        except Exception as e:
            self._logger.error("Failed to execute write query: %s", e)
            raise RuntimeError(f"Write query execution failed: {e}")

    def execute_transaction(self, transaction_func, *args, **kwargs) -> Any:
        """
        Execute a transaction.

        Args:
            transaction_func: Function to execute in transaction
            *args: Arguments for transaction function
            **kwargs: Keyword arguments for transaction function

        Returns:
            Transaction result

        Raises:
            RuntimeError: If transaction execution fails
        """
        try:
            if not self._driver:
                raise RuntimeError("Not connected to Neo4j")

            with self._driver.session(database=self._database) as session:
                result = session.execute_write(
                    transaction_func, *args, **kwargs)

                self._logger.debug("Executed transaction successfully")
                return result

        except Exception as e:
            self._logger.error("Failed to execute transaction: %s", e)
            raise RuntimeError(f"Transaction execution failed: {e}")

    def get_driver(self) -> Optional[Driver]:
        """
        Get the underlying Neo4j driver.

        Returns:
            Neo4j driver instance or None if not connected
        """
        return self._driver

    def is_connected(self) -> bool:
        """
        Check if connected to the storage backend.

        Returns:
            True if connected, False otherwise
        """
        try:
            if not self._driver:
                return False

            # Test connectivity with a simple query
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1")
                return True

        except Exception as e:
            self._logger.debug("Connection check failed: %s", e)
            return False

    def create_session(self) -> Session:
        """
        Create a new database session.

        Returns:
            Neo4j session

        Raises:
            RuntimeError: If not connected
        """
        if not self._driver:
            raise RuntimeError("Not connected to Neo4j")

        return self._driver.session(database=self._database)

    def create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes."""
        try:
            constraints_and_indexes = [
                # Fact node constraints
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",

                # Cluster node constraints
                "CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE c.id IS UNIQUE",

                # Source node constraints
                "CREATE CONSTRAINT source_url_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",

                # Indexes for performance
                "CREATE INDEX fact_content_index IF NOT EXISTS FOR (f:Fact) ON (f.content)",
                "CREATE INDEX fact_domain_index IF NOT EXISTS FOR (f:Fact) ON (f.domain)",
                "CREATE INDEX cluster_topic_index IF NOT EXISTS FOR (c:Cluster) ON (c.topic)",
                "CREATE INDEX source_domain_index IF NOT EXISTS FOR (s:Source) ON (s.domain)"
            ]

            for constraint_or_index in constraints_and_indexes:
                try:
                    self.execute_write_query(constraint_or_index)
                    self._logger.debug(
                        "Created constraint/index: %s", constraint_or_index)
                except Exception as e:
                    # Constraint/index might already exist
                    self._logger.debug(
                        "Constraint/index creation skipped: %s", e)

            self._logger.info("Constraints and indexes setup completed")

        except Exception as e:
            self._logger.error(
                "Failed to create constraints and indexes: %s", e)
            raise

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information.

        Returns:
            Database information dictionary
        """
        try:
            # Get node counts
            node_counts = self.execute_query("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
            """)

            # Get relationship counts
            rel_counts = self.execute_query("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """)

            # Get database version
            version_result = self.execute_query("CALL dbms.components()")
            version = version_result[0]['versions'][0] if version_result else "Unknown"

            return {
                'version': version,
                'database': self._database,
                'node_counts': {str(item['labels']): item['count'] for item in node_counts},
                'relationship_counts': {item['type']: item['count'] for item in rel_counts}
            }

        except Exception as e:
            self._logger.error("Failed to get database info: %s", e)
            return {'error': str(e)}

    def dispose(self) -> None:
        """Dispose of resources."""
        self.disconnect()
        self._logger.info("Neo4j storage provider disposed")
