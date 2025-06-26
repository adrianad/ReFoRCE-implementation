"""
SQL execution and validation utilities for ReFoRCE
"""
import logging
import asyncio
import sqlparse
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of SQL query execution"""
    success: bool
    data: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    execution_time: float
    rows_affected: int
    query_hash: str

@dataclass
class ValidationResult:
    """Result of SQL validation"""
    is_valid: bool
    syntax_errors: List[str]
    semantic_errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class SQLExecutor:
    """Handles SQL execution, validation, and feedback generation"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.execution_history = []
        self.query_cache = {}
    
    def validate_sql_syntax(self, sql: str) -> ValidationResult:
        """
        Validate SQL syntax using sqlparse and database-specific checks
        """
        syntax_errors = []
        semantic_errors = []
        warnings = []
        suggestions = []
        
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                syntax_errors.append("Empty or invalid SQL statement")
                return ValidationResult(False, syntax_errors, semantic_errors, warnings, suggestions)
            
            # Basic syntax validation
            statement = parsed[0]
            
            # Check for common syntax issues
            sql_upper = sql.upper().strip()
            
            # Check for SQL injection patterns (basic)
            dangerous_patterns = [
                "DROP TABLE", "DELETE FROM", "TRUNCATE", "ALTER TABLE",
                "CREATE USER", "GRANT", "REVOKE"
            ]
            
            for pattern in dangerous_patterns:
                if pattern in sql_upper:
                    warnings.append(f"Potentially dangerous operation detected: {pattern}")
            
            # Check for basic SELECT structure
            if sql_upper.startswith("SELECT"):
                if "FROM" not in sql_upper:
                    warnings.append("SELECT statement without FROM clause - may return limited results")
            
            # Database-specific validation
            db_valid, db_error = self.db_manager.validate_sql_syntax(sql)
            if not db_valid:
                syntax_errors.append(f"Database validation failed: {db_error}")
            
            # Performance suggestions
            if "SELECT *" in sql_upper:
                suggestions.append("Consider specifying column names instead of SELECT *")
            
            if sql_upper.count("JOIN") > 5:
                suggestions.append("Complex query with many JOINs - consider breaking into subqueries")
            
            is_valid = len(syntax_errors) == 0
            
            return ValidationResult(is_valid, syntax_errors, semantic_errors, warnings, suggestions)
            
        except Exception as e:
            syntax_errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, syntax_errors, semantic_errors, warnings, suggestions)
    
    async def execute_sql_safe(self, sql: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute SQL with safety checks and timeout
        """
        import time
        import hashlib
        
        # Generate query hash for caching
        query_hash = hashlib.md5(sql.encode()).hexdigest()
        
        # Check cache first
        if query_hash in self.query_cache:
            cached_result = self.query_cache[query_hash]
            logger.info(f"Returning cached result for query {query_hash[:8]}...")
            return cached_result
        
        start_time = time.time()
        
        try:
            # Validate before execution
            validation = self.validate_sql_syntax(sql)
            if not validation.is_valid:
                error_msg = f"SQL validation failed: {'; '.join(validation.syntax_errors)}"
                result = ExecutionResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    execution_time=time.time() - start_time,
                    rows_affected=0,
                    query_hash=query_hash
                )
                return result
            
            # Execute with timeout
            data = await self.db_manager.execute_with_timeout(sql, timeout)
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                data=data,
                error=None,
                execution_time=execution_time,
                rows_affected=len(data) if data else 0,
                query_hash=query_hash
            )
            
            # Cache successful results
            self.query_cache[query_hash] = result
            
            # Log execution
            self.execution_history.append({
                "timestamp": start_time,
                "sql": sql,
                "success": True,
                "execution_time": execution_time,
                "rows_returned": len(data) if data else 0
            })
            
            logger.info(f"SQL executed successfully: {len(data) if data else 0} rows in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Query timed out after {timeout} seconds"
            result = ExecutionResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=timeout,
                rows_affected=0,
                query_hash=query_hash
            )
            
            self.execution_history.append({
                "timestamp": start_time,
                "sql": sql,
                "success": False,
                "error": error_msg,
                "execution_time": timeout
            })
            
            logger.error(error_msg)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            result = ExecutionResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time,
                rows_affected=0,
                query_hash=query_hash
            )
            
            self.execution_history.append({
                "timestamp": start_time,
                "sql": sql,
                "success": False,
                "error": error_msg,
                "execution_time": execution_time
            })
            
            logger.error(f"SQL execution failed: {error_msg}")
            return result
    
    def generate_execution_feedback(self, sql: str, result: ExecutionResult) -> str:
        """
        Generate detailed feedback about SQL execution results
        """
        feedback_parts = []
        
        if result.success:
            feedback_parts.append(f"✓ Query executed successfully")
            feedback_parts.append(f"✓ Returned {result.rows_affected} rows")
            feedback_parts.append(f"✓ Execution time: {result.execution_time:.2f} seconds")
            
            # Performance feedback
            if result.execution_time > 5.0:
                feedback_parts.append("⚠ Query took longer than 5 seconds - consider optimization")
            
            if result.rows_affected > 10000:
                feedback_parts.append("⚠ Large result set - consider adding LIMIT clause")
            
            # Data type feedback
            if result.data and len(result.data) > 0:
                sample_row = result.data[0]
                feedback_parts.append(f"✓ Sample columns: {', '.join(sample_row.keys())}")
                
                # Check for null values
                null_columns = [k for k, v in sample_row.items() if v is None]
                if null_columns:
                    feedback_parts.append(f"ℹ Columns with null values: {', '.join(null_columns)}")
        
        else:
            feedback_parts.append(f"✗ Query execution failed")
            feedback_parts.append(f"✗ Error: {result.error}")
            
            # Analyze error for suggestions
            error_lower = result.error.lower()
            
            if "column" in error_lower and "does not exist" in error_lower:
                feedback_parts.append("💡 Suggestion: Check column names in your query against the database schema")
            
            if "table" in error_lower and "does not exist" in error_lower:
                feedback_parts.append("💡 Suggestion: Verify table names are correct and exist in the database")
            
            if "syntax error" in error_lower:
                feedback_parts.append("💡 Suggestion: Review SQL syntax - check for missing commas, parentheses, or keywords")
            
            if "permission denied" in error_lower:
                feedback_parts.append("💡 Suggestion: Check database permissions for the operation")
            
            if "timeout" in error_lower:
                feedback_parts.append("💡 Suggestion: Optimize query or add WHERE clauses to limit data processing")
        
        return "\n".join(feedback_parts)
    
    def analyze_query_performance(self, sql: str) -> Dict[str, Any]:
        """
        Analyze query for potential performance issues
        """
        analysis = {
            "complexity_score": 0,
            "performance_warnings": [],
            "optimization_suggestions": []
        }
        
        sql_upper = sql.upper()
        
        # Count complexity factors
        join_count = sql_upper.count("JOIN")
        subquery_count = sql_upper.count("SELECT") - 1  # Subtract main SELECT
        where_conditions = sql_upper.count("WHERE")
        
        analysis["complexity_score"] = join_count * 2 + subquery_count * 3 + where_conditions
        
        # Performance warnings
        if "SELECT *" in sql_upper:
            analysis["performance_warnings"].append("Using SELECT * may retrieve unnecessary columns")
        
        if join_count > 3:
            analysis["performance_warnings"].append(f"Query has {join_count} JOINs - may be complex")
        
        if subquery_count > 2:
            analysis["performance_warnings"].append(f"Query has {subquery_count} subqueries - consider CTEs")
        
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            analysis["performance_warnings"].append("ORDER BY without LIMIT may sort entire result set")
        
        # Optimization suggestions
        if "DISTINCT" in sql_upper:
            analysis["optimization_suggestions"].append("Consider if DISTINCT is necessary - it adds sorting overhead")
        
        if where_conditions == 0 and "SELECT" in sql_upper:
            analysis["optimization_suggestions"].append("Consider adding WHERE clause to limit results")
        
        return analysis
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about SQL execution history
        """
        if not self.execution_history:
            return {"total_queries": 0}
        
        successful_queries = [q for q in self.execution_history if q["success"]]
        failed_queries = [q for q in self.execution_history if not q["success"]]
        
        avg_execution_time = 0
        if successful_queries:
            avg_execution_time = sum(q["execution_time"] for q in successful_queries) / len(successful_queries)
        
        return {
            "total_queries": len(self.execution_history),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": len(successful_queries) / len(self.execution_history) if self.execution_history else 0,
            "average_execution_time": avg_execution_time,
            "cache_hits": len(self.query_cache),
            "recent_errors": [q["error"] for q in failed_queries[-5:]]  # Last 5 errors
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_schema_suggestions(self, error_message: str) -> List[str]:
        """
        Generate schema-based suggestions for fixing errors
        """
        suggestions = []
        
        try:
            if "column" in error_message.lower():
                # Get available tables and columns
                tables = self.db_manager.get_all_tables()
                suggestions.append(f"Available tables: {', '.join(tables[:10])}")
                
                # Try to extract mentioned column name
                import re
                column_match = re.search(r'column "([^"]+)"', error_message)
                if column_match:
                    column_name = column_match.group(1)
                    suggestions.append(f"Column '{column_name}' not found - check spelling and table reference")
            
            if "table" in error_message.lower():
                tables = self.db_manager.get_all_tables()
                suggestions.append(f"Available tables: {', '.join(tables[:10])}")
                
                table_match = re.search(r'table "([^"]+)"', error_message)
                if table_match:
                    table_name = table_match.group(1)
                    similar_tables = [t for t in tables if table_name.lower() in t.lower()]
                    if similar_tables:
                        suggestions.append(f"Similar table names: {', '.join(similar_tables[:5])}")
        
        except Exception as e:
            logger.warning(f"Could not generate schema suggestions: {e}")
        
        return suggestions