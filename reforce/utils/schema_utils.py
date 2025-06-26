"""
Schema utilities for ReFoRCE Text-to-SQL system
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    columns: List[Dict[str, Any]]
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    relationships: List[str] = None

@dataclass
class ColumnInfo:
    """Information about a database column"""
    name: str
    table: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None

class SchemaUtils:
    """Utility functions for schema analysis and manipulation"""
    
    @staticmethod
    def parse_ddl_statement(ddl: str) -> TableInfo:
        """Parse DDL statement to extract table information"""
        try:
            # Extract table name
            table_match = re.search(r'CREATE TABLE\s+(\w+)', ddl, re.IGNORECASE)
            if not table_match:
                raise ValueError("Could not find table name in DDL")
            
            table_name = table_match.group(1)
            
            # Extract columns
            columns = SchemaUtils._extract_columns_from_ddl(ddl)
            
            return TableInfo(
                name=table_name,
                columns=columns,
                relationships=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to parse DDL: {e}")
            raise
    
    @staticmethod
    def _extract_columns_from_ddl(ddl: str) -> List[Dict[str, Any]]:
        """Extract column definitions from DDL"""
        columns = []
        
        # Find column definitions between parentheses
        match = re.search(r'\((.*)\)', ddl, re.DOTALL)
        if not match:
            return columns
        
        column_section = match.group(1)
        
        # Split by commas (but not within parentheses)
        column_lines = SchemaUtils._split_column_definitions(column_section)
        
        for line in column_lines:
            line = line.strip()
            if not line or line.upper().startswith(('CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'INDEX')):
                continue
            
            column_info = SchemaUtils._parse_column_definition(line)
            if column_info:
                columns.append(column_info)
        
        return columns
    
    @staticmethod
    def _split_column_definitions(text: str) -> List[str]:
        """Split column definitions by comma, respecting parentheses"""
        parts = []
        current_part = ""
        paren_depth = 0
        
        for char in text:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            
            current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    @staticmethod
    def _parse_column_definition(definition: str) -> Optional[Dict[str, Any]]:
        """Parse a single column definition"""
        try:
            # Basic pattern: column_name data_type [constraints]
            parts = definition.split()
            if len(parts) < 2:
                return None
            
            column_name = parts[0]
            data_type = parts[1]
            
            # Extract constraints
            definition_upper = definition.upper()
            is_nullable = 'NOT NULL' not in definition_upper
            is_primary_key = 'PRIMARY KEY' in definition_upper
            
            # Extract default value
            default_value = None
            default_match = re.search(r'DEFAULT\s+([^,\s]+)', definition, re.IGNORECASE)
            if default_match:
                default_value = default_match.group(1)
            
            return {
                'name': column_name,
                'data_type': data_type,
                'is_nullable': is_nullable,
                'is_primary_key': is_primary_key,
                'default_value': default_value
            }
            
        except Exception as e:
            logger.warning(f"Could not parse column definition '{definition}': {e}")
            return None
    
    @staticmethod
    def identify_foreign_keys(tables: List[TableInfo]) -> Dict[str, List[str]]:
        """Identify potential foreign key relationships between tables"""
        relationships = {}
        
        # Build column index
        all_columns = {}
        for table in tables:
            for column in table.columns:
                col_name = column['name'].lower()
                if col_name not in all_columns:
                    all_columns[col_name] = []
                all_columns[col_name].append(f"{table.name}.{column['name']}")
        
        # Look for naming patterns that suggest relationships
        for table in tables:
            table_relationships = []
            
            for column in table.columns:
                col_name = column['name'].lower()
                
                # Pattern 1: column ends with _id
                if col_name.endswith('_id'):
                    base_name = col_name[:-3]  # Remove _id
                    # Look for tables with similar names
                    for other_table in tables:
                        if other_table.name.lower() == base_name or base_name in other_table.name.lower():
                            relationship = f"{table.name}.{column['name']} -> {other_table.name}"
                            table_relationships.append(relationship)
                
                # Pattern 2: column name matches table name
                for other_table in tables:
                    if col_name == other_table.name.lower():
                        relationship = f"{table.name}.{column['name']} -> {other_table.name}"
                        table_relationships.append(relationship)
            
            if table_relationships:
                relationships[table.name] = table_relationships
        
        return relationships
    
    @staticmethod
    def analyze_schema_complexity(tables: List[TableInfo]) -> Dict[str, Any]:
        """Analyze schema complexity metrics"""
        total_tables = len(tables)
        total_columns = sum(len(table.columns) for table in tables)
        
        # Column distribution
        columns_per_table = [len(table.columns) for table in tables]
        avg_columns = sum(columns_per_table) / len(columns_per_table) if columns_per_table else 0
        max_columns = max(columns_per_table) if columns_per_table else 0
        
        # Data type distribution
        data_types = {}
        for table in tables:
            for column in table.columns:
                dtype = column.get('data_type', 'unknown').lower()
                data_types[dtype] = data_types.get(dtype, 0) + 1
        
        # Naming pattern analysis
        naming_patterns = SchemaUtils._analyze_naming_patterns(tables)
        
        return {
            'total_tables': total_tables,
            'total_columns': total_columns,
            'avg_columns_per_table': avg_columns,
            'max_columns_per_table': max_columns,
            'data_type_distribution': data_types,
            'naming_patterns': naming_patterns,
            'complexity_score': SchemaUtils._calculate_complexity_score(tables)
        }
    
    @staticmethod
    def _analyze_naming_patterns(tables: List[TableInfo]) -> Dict[str, int]:
        """Analyze naming patterns in table and column names"""
        patterns = {
            'snake_case': 0,
            'camelCase': 0,
            'PascalCase': 0,
            'lowercase': 0,
            'UPPERCASE': 0,
            'prefixed': 0,
            'suffixed': 0
        }
        
        all_names = []
        
        # Collect all table and column names
        for table in tables:
            all_names.append(table.name)
            for column in table.columns:
                all_names.append(column['name'])
        
        # Analyze patterns
        for name in all_names:
            if '_' in name:
                patterns['snake_case'] += 1
            elif name.islower():
                patterns['lowercase'] += 1
            elif name.isupper():
                patterns['UPPERCASE'] += 1
            elif name[0].isupper() and '_' not in name:
                patterns['PascalCase'] += 1
            elif name[0].islower() and any(c.isupper() for c in name[1:]):
                patterns['camelCase'] += 1
        
        return patterns
    
    @staticmethod
    def _calculate_complexity_score(tables: List[TableInfo]) -> float:
        """Calculate overall schema complexity score (0-10)"""
        if not tables:
            return 0.0
        
        score = 0.0
        
        # Base complexity from table count
        score += min(3.0, len(tables) / 10.0)
        
        # Column complexity
        total_columns = sum(len(table.columns) for table in tables)
        score += min(3.0, total_columns / 100.0)
        
        # Table size variation
        column_counts = [len(table.columns) for table in tables]
        if column_counts:
            avg_cols = sum(column_counts) / len(column_counts)
            max_cols = max(column_counts)
            if avg_cols > 0:
                variation = max_cols / avg_cols
                score += min(2.0, variation / 5.0)
        
        # Data type diversity
        all_types = set()
        for table in tables:
            for column in table.columns:
                all_types.add(column.get('data_type', 'unknown').lower())
        score += min(2.0, len(all_types) / 10.0)
        
        return min(10.0, score)
    
    @staticmethod
    def suggest_query_optimizations(sql: str, schema_info: List[TableInfo]) -> List[str]:
        """Suggest optimizations for SQL query based on schema"""
        suggestions = []
        sql_upper = sql.upper()
        
        # Check for SELECT *
        if 'SELECT *' in sql_upper:
            suggestions.append("Consider specifying column names instead of SELECT * for better performance")
        
        # Check for missing WHERE clauses on large tables
        for table in schema_info:
            if table.name.upper() in sql_upper and 'WHERE' not in sql_upper:
                suggestions.append(f"Consider adding WHERE clause when querying table {table.name}")
        
        # Check for Cartesian products
        join_count = sql_upper.count('JOIN')
        from_tables = len(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        
        if from_tables > 1 and join_count == 0:
            suggestions.append("Potential Cartesian product detected - consider adding explicit JOIN conditions")
        
        # Check for complex aggregations without indexes
        if 'GROUP BY' in sql_upper and 'ORDER BY' in sql_upper:
            suggestions.append("Complex aggregation with sorting - ensure appropriate indexes exist")
        
        return suggestions
    
    @staticmethod
    def extract_table_relationships_from_sql(sql: str) -> List[Tuple[str, str]]:
        """Extract table relationships from SQL JOIN conditions"""
        relationships = []
        
        # Find JOIN conditions
        join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            table1, col1, table2, col2 = match
            relationship = (f"{table1}.{col1}", f"{table2}.{col2}")
            relationships.append(relationship)
        
        return relationships
    
    @staticmethod
    def validate_column_references(sql: str, schema_info: List[TableInfo]) -> List[str]:
        """Validate that column references in SQL exist in schema"""
        errors = []
        
        # Build column lookup
        available_columns = {}
        for table in schema_info:
            table_name = table.name.lower()
            available_columns[table_name] = [col['name'].lower() for col in table.columns]
        
        # Extract column references from SQL
        # Pattern: table.column
        column_refs = re.findall(r'(\w+)\.(\w+)', sql, re.IGNORECASE)
        
        for table_name, column_name in column_refs:
            table_lower = table_name.lower()
            column_lower = column_name.lower()
            
            if table_lower not in available_columns:
                errors.append(f"Table '{table_name}' not found in schema")
            elif column_lower not in available_columns[table_lower]:
                errors.append(f"Column '{column_name}' not found in table '{table_name}'")
        
        return errors
    
    @staticmethod
    def generate_schema_summary(tables: List[TableInfo]) -> str:
        """Generate a human-readable schema summary"""
        if not tables:
            return "No tables found in schema"
        
        summary_parts = [
            f"Database Schema Summary:",
            f"========================",
            f"Total Tables: {len(tables)}",
            f"Total Columns: {sum(len(table.columns) for table in tables)}",
            "",
            "Tables:"
        ]
        
        for table in tables[:20]:  # Limit to first 20 tables
            column_info = []
            for col in table.columns[:5]:  # Limit to first 5 columns
                col_desc = f"{col['name']} ({col.get('data_type', 'unknown')})"
                if not col.get('is_nullable', True):
                    col_desc += " NOT NULL"
                if col.get('is_primary_key', False):
                    col_desc += " PK"
                column_info.append(col_desc)
            
            more_cols = ""
            if len(table.columns) > 5:
                more_cols = f" ... and {len(table.columns) - 5} more"
            
            summary_parts.append(f"  • {table.name}: {', '.join(column_info)}{more_cols}")
        
        if len(tables) > 20:
            summary_parts.append(f"  ... and {len(tables) - 20} more tables")
        
        return "\n".join(summary_parts)