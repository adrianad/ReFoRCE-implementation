"""
Validation utilities for ReFoRCE Text-to-SQL system
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    level: ValidationLevel
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    issues: List[ValidationIssue]
    confidence_score: float

class SQLValidator:
    """Comprehensive SQL validation utilities"""
    
    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        r'DROP\s+TABLE',
        r'DELETE\s+FROM',
        r'TRUNCATE\s+TABLE?',
        r'ALTER\s+TABLE',
        r'CREATE\s+USER',
        r'GRANT\s+',
        r'REVOKE\s+',
        r'--\s*',  # SQL comments can hide malicious code
        r'/\*.*\*/',  # Block comments
    ]
    
    # Required keywords for different query types
    REQUIRED_KEYWORDS = {
        'SELECT': ['FROM'],
        'INSERT': ['INTO', 'VALUES'],
        'UPDATE': ['SET'],
        'DELETE': ['FROM']
    }
    
    @staticmethod
    def validate_sql_security(sql: str) -> ValidationResult:
        """Validate SQL for security issues"""
        issues = []
        
        # Check for dangerous patterns
        for pattern in SQLValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Potentially dangerous SQL pattern detected: {pattern}",
                    suggestion="Remove or replace dangerous operations"
                ))
        
        # Check for SQL injection patterns
        injection_patterns = [
            r"'\s*OR\s*'",  # '1' OR '1'
            r"'\s*;\s*",    # '; DROP TABLE
            r"UNION\s+SELECT",  # UNION based injection
            r"--\s*\w+",    # Comment injection
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Potential SQL injection pattern: {pattern}",
                    suggestion="Validate and sanitize input parameters"
                ))
        
        is_valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
        confidence = 1.0 - (len(issues) * 0.2)
        
        return ValidationResult(is_valid, issues, max(0.0, confidence))
    
    @staticmethod
    def validate_sql_syntax(sql: str) -> ValidationResult:
        """Validate basic SQL syntax"""
        issues = []
        sql_trimmed = sql.strip()
        
        if not sql_trimmed:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Empty SQL query",
                suggestion="Provide a valid SQL statement"
            ))
            return ValidationResult(False, issues, 0.0)
        
        # Check for basic SQL structure
        sql_upper = sql_trimmed.upper()
        
        # Identify query type
        query_type = None
        for qtype in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
            if sql_upper.startswith(qtype):
                query_type = qtype
                break
        
        if not query_type:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Unrecognized SQL query type",
                suggestion="Start with SELECT, INSERT, UPDATE, or DELETE"
            ))
        else:
            # Check required keywords
            required = SQLValidator.REQUIRED_KEYWORDS.get(query_type, [])
            for keyword in required:
                if keyword not in sql_upper:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"{query_type} query missing required keyword: {keyword}",
                        suggestion=f"Add {keyword} clause to complete the query"
                    ))
        
        # Check parentheses balance
        if sql.count('(') != sql.count(')'):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Unbalanced parentheses",
                suggestion="Check opening and closing parentheses"
            ))
        
        # Check quote balance
        single_quotes = sql.count("'") - sql.count("\\'")  # Exclude escaped quotes
        if single_quotes % 2 != 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Unbalanced single quotes",
                suggestion="Check string literals for missing quotes"
            ))
        
        # Check for common syntax errors
        if re.search(r'SELECT\s*,', sql, re.IGNORECASE):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="SELECT with leading comma",
                suggestion="Remove leading comma after SELECT"
            ))
        
        if re.search(r',\s*FROM', sql, re.IGNORECASE):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Trailing comma before FROM",
                suggestion="Remove trailing comma in SELECT clause"
            ))
        
        is_valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
        confidence = 1.0 - (len([i for i in issues if i.level == ValidationLevel.ERROR]) * 0.3)
        
        return ValidationResult(is_valid, issues, max(0.0, confidence))
    
    @staticmethod
    def validate_sql_performance(sql: str) -> ValidationResult:
        """Validate SQL for performance issues"""
        issues = []
        sql_upper = sql.upper()
        
        # Check for SELECT *
        if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Using SELECT * may retrieve unnecessary columns",
                suggestion="Specify only required columns for better performance"
            ))
        
        # Check for missing WHERE clauses
        if 'SELECT' in sql_upper and 'FROM' in sql_upper and 'WHERE' not in sql_upper:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Query without WHERE clause may return all rows",
                suggestion="Consider adding WHERE clause to limit results"
            ))
        
        # Check for complex JOINs without limits
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        if join_count > 3 and 'LIMIT' not in sql_upper:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message=f"Complex query with {join_count} JOINs without LIMIT",
                suggestion="Consider adding LIMIT clause for large result sets"
            ))
        
        # Check for functions in WHERE clauses
        where_functions = re.findall(r'WHERE.*?\b(\w+)\s*\([^)]*\)', sql, re.IGNORECASE | re.DOTALL)
        if where_functions:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message=f"Functions in WHERE clause may prevent index usage: {', '.join(where_functions)}",
                suggestion="Consider restructuring to use indexes effectively"
            ))
        
        # Check for ORDER BY without LIMIT
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message="ORDER BY without LIMIT may sort entire result set",
                suggestion="Add LIMIT clause if you don't need all sorted results"
            ))
        
        confidence = 1.0 - (len(issues) * 0.1)
        return ValidationResult(True, issues, max(0.0, confidence))
    
    @staticmethod
    def validate_column_references(sql: str, available_tables: Dict[str, List[str]]) -> ValidationResult:
        """Validate that column references exist in schema"""
        issues = []
        
        # Extract table.column references
        column_refs = re.findall(r'(\w+)\.(\w+)', sql, re.IGNORECASE)
        
        for table_name, column_name in column_refs:
            table_lower = table_name.lower()
            column_lower = column_name.lower()
            
            # Check if table exists
            table_found = False
            for available_table in available_tables.keys():
                if available_table.lower() == table_lower:
                    table_found = True
                    # Check if column exists in table
                    available_columns = [col.lower() for col in available_tables[available_table]]
                    if column_lower not in available_columns:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.ERROR,
                            message=f"Column '{column_name}' not found in table '{table_name}'",
                            location=f"{table_name}.{column_name}",
                            suggestion=f"Available columns: {', '.join(available_tables[available_table][:5])}"
                        ))
                    break
            
            if not table_found:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Table '{table_name}' not found in schema",
                    location=table_name,
                    suggestion=f"Available tables: {', '.join(list(available_tables.keys())[:5])}"
                ))
        
        # Extract unqualified column references in SELECT
        select_columns = re.findall(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_columns:
            columns_text = select_columns[0]
            unqualified_columns = re.findall(r'\b(\w+)\b(?!\s*\.)', columns_text)
            
            # Filter out SQL keywords
            sql_keywords = {'DISTINCT', 'AS', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'FROM'}
            unqualified_columns = [col for col in unqualified_columns 
                                 if col.upper() not in sql_keywords and not col.isdigit()]
            
            if len(unqualified_columns) > 2:  # Allow some unqualified references
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Many unqualified column references - consider using table aliases",
                    suggestion="Use table.column notation for clarity"
                ))
        
        is_valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
        confidence = 1.0 - (len([i for i in issues if i.level == ValidationLevel.ERROR]) * 0.4)
        
        return ValidationResult(is_valid, issues, max(0.0, confidence))

class RequestValidator:
    """Validate natural language requests"""
    
    @staticmethod
    def validate_request_clarity(request: str) -> ValidationResult:
        """Validate clarity and completeness of natural language request"""
        issues = []
        
        request_lower = request.lower().strip()
        
        if len(request_lower) < 10:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Request is very short and may lack detail",
                suggestion="Provide more specific details about what data you need"
            ))
        
        # Check for SQL keywords in natural language (may indicate confusion)
        sql_keywords_in_request = ['select', 'from', 'where', 'join', 'group by', 'order by']
        found_keywords = [kw for kw in sql_keywords_in_request if kw in request_lower]
        
        if found_keywords:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message=f"SQL keywords found in natural language request: {', '.join(found_keywords)}",
                suggestion="Consider using more natural language or provide SQL directly"
            ))
        
        # Check for question words (good indicators of clear requests)
        question_words = ['what', 'how', 'when', 'where', 'who', 'which', 'show', 'find', 'get', 'list']
        has_question_word = any(word in request_lower for word in question_words)
        
        if not has_question_word:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message="Request doesn't contain clear question words",
                suggestion="Start with words like 'show', 'find', 'what', etc."
            ))
        
        # Check for specificity indicators
        specificity_indicators = ['all', 'total', 'count', 'sum', 'average', 'maximum', 'minimum', 'last', 'first']
        has_specificity = any(indicator in request_lower for indicator in specificity_indicators)
        
        if not has_specificity:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                message="Request may benefit from more specific requirements",
                suggestion="Specify if you want counts, totals, specific records, etc."
            ))
        
        confidence = 1.0 - (len(issues) * 0.1)
        return ValidationResult(True, issues, max(0.0, confidence))

class ComprehensiveValidator:
    """Main validator that combines all validation types"""
    
    @staticmethod
    def validate_complete_pipeline(
        natural_request: str,
        generated_sql: str,
        available_tables: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, ValidationResult]:
        """Perform comprehensive validation of the entire pipeline"""
        
        results = {}
        
        # Validate natural language request
        results['request'] = RequestValidator.validate_request_clarity(natural_request)
        
        # Validate SQL security
        results['security'] = SQLValidator.validate_sql_security(generated_sql)
        
        # Validate SQL syntax
        results['syntax'] = SQLValidator.validate_sql_syntax(generated_sql)
        
        # Validate SQL performance
        results['performance'] = SQLValidator.validate_sql_performance(generated_sql)
        
        # Validate column references if schema available
        if available_tables:
            results['schema'] = SQLValidator.validate_column_references(generated_sql, available_tables)
        
        return results
    
    @staticmethod
    def get_overall_confidence(validation_results: Dict[str, ValidationResult]) -> float:
        """Calculate overall confidence from all validation results"""
        if not validation_results:
            return 0.0
        
        # Weight different validation types
        weights = {
            'security': 0.3,
            'syntax': 0.3,
            'schema': 0.2,
            'performance': 0.1,
            'request': 0.1
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for validation_type, result in validation_results.items():
            weight = weights.get(validation_type, 0.1)
            weighted_confidence += result.confidence_score * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def summarize_validation_issues(validation_results: Dict[str, ValidationResult]) -> Dict[str, List[str]]:
        """Summarize all validation issues by severity"""
        summary = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        for validation_type, result in validation_results.items():
            for issue in result.issues:
                issue_text = f"[{validation_type}] {issue.message}"
                if issue.suggestion:
                    issue_text += f" → {issue.suggestion}"
                
                if issue.level == ValidationLevel.ERROR:
                    summary['errors'].append(issue_text)
                elif issue.level == ValidationLevel.WARNING:
                    summary['warnings'].append(issue_text)
                else:
                    summary['info'].append(issue_text)
        
        return summary
    
    @staticmethod
    def is_safe_for_execution(validation_results: Dict[str, ValidationResult]) -> bool:
        """Determine if SQL is safe for execution based on validation"""
        
        # Check for security errors
        security_result = validation_results.get('security')
        if security_result and not security_result.is_valid:
            return False
        
        # Check for syntax errors
        syntax_result = validation_results.get('syntax')
        if syntax_result and not syntax_result.is_valid:
            return False
        
        # Check for schema errors
        schema_result = validation_results.get('schema')
        if schema_result and not schema_result.is_valid:
            return False
        
        return True