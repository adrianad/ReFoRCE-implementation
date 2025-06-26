"""
Database schema compression algorithm implementing ReFoRCE's 96% reduction technique
Uses pattern-based table grouping and DDL compression
"""
import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
from .database_manager import DatabaseManager
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TableGroup:
    """Represents a group of tables with similar patterns"""
    pattern: str
    tables: List[str]
    representative_table: str
    total_ddl_size: int
    compressed_ddl: str
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio for this group"""
        if self.total_ddl_size == 0:
            return 0.0
        return (self.total_ddl_size - len(self.compressed_ddl)) / self.total_ddl_size

class SchemaCompressor:
    """Implements ReFoRCE database information compression algorithm"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.config = settings.reforce
        
    def extract_table_patterns(self, tables: List[str]) -> Dict[str, List[str]]:
        """
        Extract common patterns from table names using prefix/suffix matching
        Returns mapping of pattern -> list of matching tables
        """
        patterns = defaultdict(list)
        
        # Pattern 1: Common prefixes (e.g., GA_SESSIONS_*, USER_DATA_*)
        prefix_groups = defaultdict(list)
        for table in tables:
            # Look for underscore-separated prefixes
            parts = table.split('_')
            if len(parts) >= 2:
                prefix = '_'.join(parts[:-1]) + '_'
                prefix_groups[prefix].append(table)
        
        # Only keep prefixes with multiple tables
        for prefix, table_list in prefix_groups.items():
            if len(table_list) > 1:
                patterns[f"prefix_{prefix}"] = table_list
        
        # Pattern 2: Common suffixes (e.g., *_LOG, *_BACKUP)
        suffix_groups = defaultdict(list)
        for table in tables:
            parts = table.split('_')
            if len(parts) >= 2:
                suffix = '_' + '_'.join(parts[1:])
                suffix_groups[suffix].append(table)
        
        # Only keep suffixes with multiple tables
        for suffix, table_list in suffix_groups.items():
            if len(table_list) > 1:
                patterns[f"suffix_{suffix}"] = table_list
        
        # Pattern 3: Date/time patterns (e.g., TABLE_20240101, LOG_2024_01)
        date_pattern = re.compile(r'.*_(\d{4,8}|\d{4}_\d{2}|\d{4}_\d{2}_\d{2})$')
        date_groups = defaultdict(list)
        for table in tables:
            if date_pattern.match(table):
                base_name = re.sub(r'_\d{4,8}|\d{4}_\d{2}|\d{4}_\d{2}_\d{2}$', '', table)
                date_groups[f"date_{base_name}"].append(table)
        
        # Only keep date patterns with multiple tables
        for pattern, table_list in date_groups.items():
            if len(table_list) > 1:
                patterns[pattern] = table_list
        
        # Pattern 4: Numeric suffixes (e.g., TABLE_1, TABLE_2)
        numeric_pattern = re.compile(r'(.+)_\d+$')
        numeric_groups = defaultdict(list)
        for table in tables:
            match = numeric_pattern.match(table)
            if match:
                base_name = match.group(1)
                numeric_groups[f"numeric_{base_name}"].append(table)
        
        # Only keep numeric patterns with multiple tables
        for pattern, table_list in numeric_groups.items():
            if len(table_list) > 1:
                patterns[pattern] = table_list
        
        return dict(patterns)
    
    def select_representative_table(self, tables: List[str]) -> str:
        """
        Select the most representative table from a group
        Prefers tables with more complete schemas or common naming
        """
        if not tables:
            return ""
        
        # Strategy 1: Prefer tables without date/numeric suffixes
        non_temporal = [t for t in tables if not re.search(r'_\d{4,8}$|_\d+$', t)]
        if non_temporal:
            return min(non_temporal)  # Choose alphabetically first
        
        # Strategy 2: Choose the table with the most columns
        table_column_counts = {}
        for table in tables[:min(5, len(tables))]:  # Limit checks for performance
            try:
                schema = self.db_manager.get_table_schema(table)
                table_column_counts[table] = len(schema)
            except Exception as e:
                logger.warning(f"Could not get schema for {table}: {e}")
                table_column_counts[table] = 0
        
        if table_column_counts:
            return max(table_column_counts, key=table_column_counts.get)
        
        # Fallback: Choose first table alphabetically
        return min(tables)
    
    def compress_table_group(self, pattern: str, tables: List[str]) -> TableGroup:
        """
        Compress a group of similar tables into a representative DDL
        """
        representative = self.select_representative_table(tables)
        
        try:
            # Get full DDL for representative table
            representative_ddl = self.db_manager.get_table_ddl(representative)
            
            # Calculate total original DDL size
            total_size = 0
            for table in tables:
                try:
                    ddl = self.db_manager.get_table_ddl(table)
                    total_size += len(ddl.encode('utf-8'))
                except Exception as e:
                    logger.warning(f"Could not get DDL for {table}: {e}")
            
            # Create compressed representation
            other_tables = [t for t in tables if t != representative]
            compressed_ddl = self._create_compressed_ddl(
                representative, representative_ddl, other_tables, pattern
            )
            
            return TableGroup(
                pattern=pattern,
                tables=tables,
                representative_table=representative,
                total_ddl_size=total_size,
                compressed_ddl=compressed_ddl
            )
            
        except Exception as e:
            logger.error(f"Failed to compress table group {pattern}: {e}")
            # Fallback to basic compression
            return TableGroup(
                pattern=pattern,
                tables=tables,
                representative_table=representative,
                total_ddl_size=sum(len(t) * 100 for t in tables),  # Estimate
                compressed_ddl=f"-- Pattern: {pattern}\n-- Tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}"
            )
    
    def _create_compressed_ddl(self, representative: str, representative_ddl: str, 
                              other_tables: List[str], pattern: str) -> str:
        """Create compressed DDL representation"""
        compressed = [
            f"-- COMPRESSED TABLE GROUP: {pattern}",
            f"-- Representative table: {representative}",
            f"-- Total tables in group: {len(other_tables) + 1}",
            "",
            "-- Full DDL for representative table:",
            representative_ddl,
            "",
            "-- Other tables in this group (schema assumed similar):"
        ]
        
        # List other tables in chunks to avoid huge strings
        chunk_size = 10
        for i in range(0, len(other_tables), chunk_size):
            chunk = other_tables[i:i + chunk_size]
            compressed.append(f"-- {', '.join(chunk)}")
        
        if len(other_tables) > 50:
            compressed.append(f"-- ... and {len(other_tables) - 50} more tables")
        
        return "\n".join(compressed)
    
    def compress_database_schema(self) -> Tuple[Dict[str, TableGroup], List[str], float]:
        """
        Main compression algorithm implementing ReFoRCE's approach
        Returns: (compressed_groups, ungrouped_tables, compression_ratio)
        """
        try:
            # Get all tables
            all_tables = self.db_manager.get_all_tables()
            logger.info(f"Found {len(all_tables)} tables to compress")
            
            if not all_tables:
                return {}, [], 0.0
            
            # Extract patterns
            patterns = self.extract_table_patterns(all_tables)
            logger.info(f"Found {len(patterns)} table patterns")
            
            # Group tables and compress
            compressed_groups = {}
            grouped_tables = set()
            
            for pattern, tables in patterns.items():
                if len(tables) >= 2:  # Only group if multiple tables
                    group = self.compress_table_group(pattern, tables)
                    compressed_groups[pattern] = group
                    grouped_tables.update(tables)
                    logger.info(f"Compressed {len(tables)} tables in pattern {pattern}")
            
            # Identify ungrouped tables
            ungrouped_tables = [t for t in all_tables if t not in grouped_tables]
            
            # Calculate overall compression ratio
            total_original_size = 0
            total_compressed_size = 0
            
            for group in compressed_groups.values():
                total_original_size += group.total_ddl_size
                total_compressed_size += len(group.compressed_ddl.encode('utf-8'))
            
            # Add size for ungrouped tables (no compression)
            for table in ungrouped_tables:
                try:
                    ddl = self.db_manager.get_table_ddl(table)
                    size = len(ddl.encode('utf-8'))
                    total_original_size += size
                    total_compressed_size += size
                except Exception as e:
                    logger.warning(f"Could not calculate size for {table}: {e}")
            
            compression_ratio = 0.0
            if total_original_size > 0:
                compression_ratio = (total_original_size - total_compressed_size) / total_original_size
            
            logger.info(f"Schema compression completed:")
            logger.info(f"  - Original size: {total_original_size:,} bytes")
            logger.info(f"  - Compressed size: {total_compressed_size:,} bytes")
            logger.info(f"  - Compression ratio: {compression_ratio:.2%}")
            logger.info(f"  - Grouped tables: {len(grouped_tables)}")
            logger.info(f"  - Ungrouped tables: {len(ungrouped_tables)}")
            
            return compressed_groups, ungrouped_tables, compression_ratio
            
        except Exception as e:
            logger.error(f"Schema compression failed: {e}")
            raise
    
    def get_compressed_schema_text(self) -> str:
        """
        Get the complete compressed schema as text for LLM consumption
        """
        if self.config.ultra_compact_mode:
            return self._get_ultra_compact_schema()
        
        compressed_groups, ungrouped_tables, compression_ratio = self.compress_database_schema()
        
        schema_parts = [
            f"-- COMPRESSED DATABASE SCHEMA (Compression ratio: {compression_ratio:.2%})",
            f"-- Total tables: {sum(len(g.tables) for g in compressed_groups.values()) + len(ungrouped_tables)}",
            f"-- Compressed groups: {len(compressed_groups)}",
            f"-- Ungrouped tables: {len(ungrouped_tables)}",
            "",
            "-- COMPRESSED TABLE GROUPS:",
            "=" * 80
        ]
        
        for pattern, group in compressed_groups.items():
            schema_parts.append("")
            schema_parts.append(group.compressed_ddl)
            schema_parts.append("=" * 80)
        
        if ungrouped_tables:
            schema_parts.extend([
                "",
                "-- INDIVIDUAL TABLES (No compression applied):",
                "=" * 80
            ])
            
            for table in ungrouped_tables:
                try:
                    ddl = self.db_manager.get_table_ddl(table)
                    schema_parts.append(f"-- Table: {table}")
                    schema_parts.append(ddl)
                    schema_parts.append("-" * 40)
                except Exception as e:
                    schema_parts.append(f"-- Table: {table} (DDL unavailable: {e})")
        
        return "\n".join(schema_parts)
    
    def get_table_groups_summary(self) -> Dict[str, Dict]:
        """Get summary of table groups for analysis"""
        compressed_groups, ungrouped_tables, compression_ratio = self.compress_database_schema()
        
        summary = {
            "compression_ratio": compression_ratio,
            "total_groups": len(compressed_groups),
            "total_ungrouped": len(ungrouped_tables),
            "groups": {}
        }
        
        for pattern, group in compressed_groups.items():
            summary["groups"][pattern] = {
                "table_count": len(group.tables),
                "representative": group.representative_table,
                "compression_ratio": group.compression_ratio(),
                "original_size": group.total_ddl_size,
                "compressed_size": len(group.compressed_ddl.encode('utf-8'))
            }
        
        return summary
    
    def _get_ultra_compact_schema(self) -> str:
        """
        Get ultra compact schema representation
        Format: table_name: col1, col2, fk_col->ref_table, col3
        """
        try:
            all_tables = self.db_manager.get_all_tables()
            compact_lines = [f"-- ULTRA COMPACT SCHEMA ({len(all_tables)} tables)"]
            
            for table in all_tables:
                try:
                    compact_table = self._get_ultra_compact_table(table)
                    if compact_table:
                        compact_lines.append(compact_table)
                except Exception as e:
                    logger.warning(f"Could not generate ultra compact format for {table}: {e}")
                    compact_lines.append(f"{table}: (schema unavailable)")
            
            return "\n".join(compact_lines)
            
        except Exception as e:
            logger.error(f"Ultra compact schema generation failed: {e}")
            return "-- Ultra compact schema unavailable"
    
    def _get_ultra_compact_table(self, table_name: str) -> str:
        """
        Get ultra compact representation of a single table
        Format: table_name: col1, col2, fk_col->ref_table, col3
        """
        try:
            schema = self.db_manager.get_table_schema(table_name)
            if not schema:
                return f"{table_name}: (no columns)"
            
            # Detect foreign keys
            foreign_keys = self._detect_foreign_keys(table_name, schema)
            
            # Build column list with FK references
            columns = []
            for col_info in schema:
                col_name = col_info['column_name']
                
                # Check if this column is a foreign key
                if col_name in foreign_keys:
                    ref_table = foreign_keys[col_name]
                    columns.append(f"{col_name}->{ref_table}")
                else:
                    columns.append(col_name)
            
            return f"{table_name}: {', '.join(columns)}"
            
        except Exception as e:
            logger.error(f"Ultra compact format failed for {table_name}: {e}")
            return f"{table_name}: (error: {str(e)})"
    
    def _detect_foreign_keys(self, table_name: str, schema: List[Dict]) -> Dict[str, str]:
        """
        Detect foreign key relationships using common patterns
        Returns dict mapping column_name -> referenced_table
        """
        foreign_keys = {}
        
        # Get all table names for reference
        try:
            all_tables = set(self.db_manager.get_all_tables())
        except Exception:
            all_tables = set()
        
        for col_info in schema:
            col_name = col_info['column_name'].lower()
            
            # Pattern 1: column_name ends with 'id' and matches table name
            # e.g., 'userid' -> 'user', 'orderid' -> 'order'
            if col_name.endswith('id') and len(col_name) > 2:
                potential_table = col_name[:-2]  # Remove 'id' suffix
                
                # Check exact match
                if potential_table in all_tables:
                    foreign_keys[col_info['column_name']] = potential_table
                    continue
                
                # Check with 's' suffix (plurals)
                if potential_table + 's' in all_tables:
                    foreign_keys[col_info['column_name']] = potential_table + 's'
                    continue
                
                # Check without 's' suffix (singular)
                if potential_table.endswith('s') and potential_table[:-1] in all_tables:
                    foreign_keys[col_info['column_name']] = potential_table[:-1]
                    continue
            
            # Pattern 2: column_name format 'table_id' or 'tableid'
            # e.g., 'user_id' -> 'user', 'storage_id' -> 'storage'
            if '_id' in col_name:
                potential_table = col_name.replace('_id', '')
                if potential_table in all_tables:
                    foreign_keys[col_info['column_name']] = potential_table
                    continue
            
            # Pattern 3: Direct table name matches
            # e.g., column 'storage' and table 'storage' exists
            if col_name in all_tables and col_name != table_name.lower():
                foreign_keys[col_info['column_name']] = col_name
                continue
            
            # Pattern 4: Check if column name is substring of any table
            for table in all_tables:
                if col_name in table.lower() and col_name != table_name.lower():
                    foreign_keys[col_info['column_name']] = table
                    break
        
        return foreign_keys