"""
Database manager for PostgreSQL connections and schema operations
"""
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from contextlib import asynccontextmanager
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from ..config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, config=None):
        self.config = config or settings.database
        self._connection_pool = None
        self._sync_connection = None
    
    async def initialize_async_pool(self, min_connections: int = 5, max_connections: int = 20):
        """Initialize async connection pool"""
        try:
            self._connection_pool = await asyncpg.create_pool(
                self.config.connection_string,
                min_size=min_connections,
                max_size=max_connections
            )
            logger.info("Async connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")
            raise
    
    def get_sync_connection(self):
        """Get synchronous connection for simple operations"""
        if not self._sync_connection or self._sync_connection.closed:
            try:
                self._sync_connection = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    cursor_factory=RealDictCursor
                )
                logger.info("Sync connection established")
            except Exception as e:
                logger.error(f"Failed to establish sync connection: {e}")
                raise
        return self._sync_connection
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async connection from pool"""
        if not self._connection_pool:
            await self.initialize_async_pool()
        
        async with self._connection_pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query asynchronously"""
        try:
            async with self.get_async_connection() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_query_sync(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query synchronously"""
        try:
            conn = self.get_sync_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Sync query execution failed: {e}")
            conn.rollback()
            raise
    
    async def execute_with_timeout(self, query: str, timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute query with timeout"""
        try:
            return await asyncio.wait_for(
                self.execute_query(query),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Query timed out after {timeout} seconds")
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get detailed schema information for a table"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            ordinal_position
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query_sync(query, (table_name,))
    
    def get_all_tables(self) -> List[str]:
        """Get list of all table names in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        result = self.execute_query_sync(query)
        return [row['table_name'] for row in result]
    
    def get_table_ddl(self, table_name: str) -> str:
        """Generate DDL statement for a table"""
        schema_info = self.get_table_schema(table_name)
        if not schema_info:
            return ""
        
        ddl_parts = [f"CREATE TABLE {table_name} ("]
        
        column_definitions = []
        for col in schema_info:
            col_def = f"  {col['column_name']} {col['data_type']}"
            
            # Add length/precision info
            if col['character_maximum_length']:
                col_def += f"({col['character_maximum_length']})"
            elif col['numeric_precision'] and col['numeric_scale']:
                col_def += f"({col['numeric_precision']},{col['numeric_scale']})"
            elif col['numeric_precision']:
                col_def += f"({col['numeric_precision']})"
            
            # Add nullability
            if col['is_nullable'] == 'NO':
                col_def += " NOT NULL"
            
            # Add default value
            if col['column_default']:
                col_def += f" DEFAULT {col['column_default']}"
            
            column_definitions.append(col_def)
        
        ddl_parts.append(",\n".join(column_definitions))
        ddl_parts.append(");")
        
        return "\n".join(ddl_parts)
    
    def get_database_size(self) -> int:
        """Get total database size in bytes"""
        query = """
        SELECT pg_database_size(current_database()) as size_bytes
        """
        result = self.execute_query_sync(query)
        return result[0]['size_bytes'] if result else 0
    
    def get_table_sizes(self) -> Dict[str, int]:
        """Get size of each table in bytes"""
        query = """
        SELECT 
            schemaname,
            tablename,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY size_bytes DESC
        """
        result = self.execute_query_sync(query)
        return {row['tablename']: row['size_bytes'] for row in result}
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL syntax without executing"""
        try:
            conn = self.get_sync_connection()
            with conn.cursor() as cursor:
                # Use EXPLAIN to validate syntax without execution
                cursor.execute(f"EXPLAIN {sql}")
                return True, "Valid SQL syntax"
        except Exception as e:
            return False, str(e)
    
    async def close_connections(self):
        """Close all connections"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("Async connection pool closed")
        
        if self._sync_connection and not self._sync_connection.closed:
            self._sync_connection.close()
            logger.info("Sync connection closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_sync_connection') and self._sync_connection and not self._sync_connection.closed:
            self._sync_connection.close()