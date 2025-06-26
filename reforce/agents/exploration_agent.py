"""
Column Exploration Agent for ReFoRCE
Stage 4: Iterative column exploration for low-confidence cases
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Sequence
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.base import Response, TaskResult
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from ..core.database_manager import DatabaseManager
from ..core.sql_executor import SQLExecutor, ExecutionResult
from ..models.llm_client import LLMClient
from ..models.prompt_templates import PromptTemplates
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ExplorationQuery:
    """Represents an exploration query with metadata"""
    sql: str
    purpose: str
    complexity_level: int
    execution_result: Optional[ExecutionResult]
    insights: List[str]

@dataclass
class ColumnInsight:
    """Insights about a column discovered through exploration"""
    table_name: str
    column_name: str
    data_type: str
    sample_values: List[Any]
    null_count: int
    unique_count: int
    relationships: List[str]
    usage_patterns: List[str]

class ExplorationAgent(AssistantAgent):
    """
    AutoGen agent for iterative column exploration
    Implements ReFoRCE Stage 4: Column Exploration for Low-Confidence Cases
    """
    
    def __init__(
        self,
        name: str = "ExplorationAgent",
        description: str = "Database column exploration specialist",
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[LLMClient] = None,
        model_client: Optional[OpenAIChatCompletionClient] = None
    ):
        # Initialize components
        self.db_manager = db_manager or DatabaseManager()
        self.llm_client = llm_client or LLMClient()
        
        # Create AutoGen model client if not provided
        if model_client is None:
            model_info = ModelInfo(
                family="qwen",
                vision=False,
                function_calling=False,
                json_output=False
            )
            model_client = OpenAIChatCompletionClient(
                model=settings.llm.model_name,
                api_key="dummy",  # vLLM doesn't require real API key
                base_url=settings.llm.base_url,
                model_info=model_info
            )
        
        # Initialize base AssistantAgent with model client
        super().__init__(name=name, description=description, model_client=model_client)
        self.sql_executor = SQLExecutor(self.db_manager)
        
        # Configuration
        self.config = settings.reforce
        self.max_exploration_queries = self.config.max_exploration_queries
        self.exploration_result_limit = self.config.exploration_result_limit
        
        # State tracking
        self.user_request = None
        self.uncertain_areas = []
        self.exploration_queries = []
        self.column_insights = {}
        self.improved_candidates = []
        self.all_candidates = []  # Store all candidates from Stage 2/3
        self.voting_uncertainty = {}  # Store voting stage uncertainty data
    
    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """Handle incoming messages for column exploration"""
        try:
            latest_message = messages[-1]
            request_content = latest_message.content
            
            logger.info(f"ExplorationAgent received request: {request_content[:100]}...")
            
            # Parse request type
            if "explore_columns" in request_content.lower():
                result = await self._explore_columns(request_content)
            elif "analyze_uncertainty" in request_content.lower():
                result = await self._analyze_uncertainty_areas(request_content)
            elif "generate_improved_sql" in request_content.lower():
                result = await self._generate_improved_sql(request_content)
            elif "get_column_insights" in request_content.lower():
                result = await self._get_column_insights()
            else:
                result = await self._handle_general_exploration_request(request_content)
            
            return Response(
                chat_message=TextMessage(content=result, source=self.name)
            )
            
        except Exception as e:
            error_msg = f"ExplorationAgent error: {str(e)}"
            logger.error(error_msg)
            return Response(
                chat_message=TextMessage(content=error_msg, source=self.name)
            )
    
    def set_exploration_context(self, user_request: str, uncertain_areas: List[str], 
                              all_candidates: List[Dict] = None, voting_uncertainty: Dict = None):
        """Set exploration context from VotingAgent with all candidates"""
        self.user_request = user_request
        self.uncertain_areas = uncertain_areas
        self.all_candidates = all_candidates or []
        self.voting_uncertainty = voting_uncertainty or {}
        logger.info(f"Exploration context set with {len(uncertain_areas)} uncertain areas and {len(self.all_candidates)} candidates")
    
    async def _explore_columns(self, request: str) -> str:
        """Perform comprehensive column exploration"""
        try:
            if not self.user_request:
                # Try to extract from request
                self.user_request = self._extract_user_request(request)
            
            if not self.uncertain_areas:
                # Identify uncertain areas automatically
                await self._identify_uncertain_areas()
            
            logger.info(f"Starting column exploration for: {self.user_request[:100]}...")
            
            # Generate exploration queries
            exploration_queries = await self._generate_exploration_queries()
            
            # Execute exploration queries progressively
            insights = await self._execute_exploration_queries(exploration_queries)
            
            # Analyze insights and generate improved SQL
            improved_sql = await self._generate_improved_sql_from_insights(insights)
            
            result = f"""Column Exploration Completed!
            
Original Request: {self.user_request}
Uncertain Areas Explored: {len(self.uncertain_areas)}
Exploration Queries Executed: {len(self.exploration_queries)}
Column Insights Discovered: {len(self.column_insights)}

Key Insights:
"""
            
            for table_col, insight in list(self.column_insights.items())[:5]:
                result += f"""
- {insight.table_name}.{insight.column_name}:
  • Data Type: {insight.data_type}
  • Sample Values: {', '.join(str(v) for v in insight.sample_values[:3])}
  • Unique Values: {insight.unique_count}
  • Relationships: {', '.join(insight.relationships[:2])}
"""
            
            if improved_sql:
                result += f"""
Improved SQL Based on Exploration:
```sql
{improved_sql}
```
"""
            
            return result
            
        except Exception as e:
            logger.error(f"Column exploration failed: {e}")
            raise
    
    async def _identify_uncertain_areas(self):
        """Automatically identify uncertain areas that need exploration"""
        try:
            # Use LLM to analyze the request and identify ambiguous parts
            analysis_prompt = f"""
            Analyze the following SQL request and identify uncertain or ambiguous areas that would benefit from database exploration:
            
            Request: {self.user_request}
            
            Consider:
            1. Ambiguous column references
            2. Unclear table relationships  
            3. Complex data transformations needed
            4. Potential data quality issues
            5. Missing context about data structure
            
            List specific areas of uncertainty:
            """
            
            response = await self.llm_client.generate_completion(
                prompt=analysis_prompt,
                system_prompt=PromptTemplates.EXPLORATION_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            # Parse uncertain areas from response
            self.uncertain_areas = self._parse_uncertain_areas(response.content)
            logger.info(f"Identified {len(self.uncertain_areas)} uncertain areas")
            
        except Exception as e:
            logger.error(f"Failed to identify uncertain areas: {e}")
            self.uncertain_areas = ["General data exploration needed"]
    
    def _parse_uncertain_areas(self, response: str) -> List[str]:
        """Parse uncertain areas from LLM response"""
        areas = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                area = line[1:].strip()
                if area:
                    areas.append(area)
        
        # Fallback if no structured list found
        if not areas:
            areas = [response.strip()]
        
        return areas[:5]  # Limit to 5 areas
    
    async def _generate_exploration_queries(self) -> List[ExplorationQuery]:
        """Generate exploration queries based on uncertain areas"""
        logger.info("Generating exploration queries...")
        
        # Get available tables
        available_tables = self.db_manager.get_all_tables()
        table_list = ', '.join(available_tables[:20])  # Limit for prompt
        
        # Generate queries using LLM
        exploration_response = await self.llm_client.generate_completion(
            prompt=PromptTemplates.get_exploration_prompt(
                user_request=self.user_request,
                table_list=table_list,
                uncertainty_areas='; '.join(self.uncertain_areas)
            ),
            system_prompt=PromptTemplates.EXPLORATION_SYSTEM_PROMPT,
            temperature=0.2
        )
        
        # Parse queries from response
        queries = self._parse_exploration_queries(exploration_response.content)
        
        # Add smart exploration queries based on LLM table relevance analysis
        smart_queries = await self._generate_smart_exploration_queries(available_tables)
        queries.extend(smart_queries)
        
        return queries[:self.max_exploration_queries]
    
    def _parse_exploration_queries(self, response: str) -> List[ExplorationQuery]:
        """Parse exploration queries from LLM response"""
        queries = []
        
        # Look for SQL code blocks
        import re
        sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        
        for i, sql in enumerate(sql_blocks):
            sql = sql.strip()
            if sql and sql.upper().startswith('SELECT'):
                # Add LIMIT if not present
                if 'LIMIT' not in sql.upper():
                    sql += f' LIMIT {self.exploration_result_limit}'
                
                queries.append(ExplorationQuery(
                    sql=sql,
                    purpose=f"Exploration query {i+1}",
                    complexity_level=self._assess_query_complexity(sql),
                    execution_result=None,
                    insights=[]
                ))
        
        return queries
    
    async def _generate_smart_exploration_queries(self, tables: List[str]) -> List[ExplorationQuery]:
        """Generate smart exploration queries using LLM to find relevant tables"""
        smart_queries = []
        
        # Use candidate-based analysis to identify relevant tables
        relevant_tables = await self._find_relevant_tables_with_candidates(tables)
        
        # Generate progressive complexity exploration queries for relevant tables
        for table in relevant_tables[:5]:  # Limit to top 5 relevant tables
            # Level 1: Basic data sampling and schema
            smart_queries.extend(self._generate_level_1_queries(table))
            
            # Level 2: Column analysis and patterns
            smart_queries.extend(self._generate_level_2_queries(table))
            
            # Level 3: Basic relationships and fuzzy matching
            smart_queries.extend(self._generate_level_3_queries(table))
            
            # Level 4: Complex analysis with JOINs (if FK relationships exist)
            smart_queries.extend(self._generate_level_4_queries(table, relevant_tables))
            
            # Level 5: Advanced analytics (if previous levels successful)
            smart_queries.extend(self._generate_level_5_queries(table))
        
        return smart_queries
    
    def _generate_level_1_queries(self, table: str) -> List[ExplorationQuery]:
        """Generate Level 1 queries: Basic sampling and schema (simplest)"""
        queries = []
        
        # Basic table sampling
        queries.append(ExplorationQuery(
            sql=f"SELECT * FROM {table} LIMIT 5",
            purpose=f"Level 1: Sample data from {table}",
            complexity_level=1,
            execution_result=None,
            insights=[]
        ))
        
        # Column schema information
        queries.append(ExplorationQuery(
            sql=f"""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = '{table}' 
            ORDER BY ordinal_position
            """,
            purpose=f"Level 1: Column structure of {table}",
            complexity_level=1,
            execution_result=None,
            insights=[]
        ))
        
        # Row count
        queries.append(ExplorationQuery(
            sql=f"SELECT COUNT(*) as row_count FROM {table}",
            purpose=f"Level 1: Row count for {table}",
            complexity_level=1,
            execution_result=None,
            insights=[]
        ))
        
        return queries
    
    def _generate_level_2_queries(self, table: str) -> List[ExplorationQuery]:
        """Generate Level 2 queries: Column analysis and patterns"""
        queries = []
        
        # Get search patterns for fuzzy matching
        patterns = self._extract_search_patterns_from_request()
        
        for pattern in patterns[:3]:  # Limit to first 3 patterns
            # Fuzzy pattern matching using %target_str% (paper methodology)
            queries.append(ExplorationQuery(
                sql=f"""
                SELECT DISTINCT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                AND (column_name ILIKE '%{pattern}%' OR column_name ILIKE '%{pattern.replace('_', '')}%')
                LIMIT {self.exploration_result_limit}
                """,
                purpose=f"Level 2: Fuzzy column matching for pattern '{pattern}' in {table}",
                complexity_level=2,
                execution_result=None,
                insights=[]
            ))
        
        # Column uniqueness analysis
        queries.append(ExplorationQuery(
            sql=f"""
            SELECT column_name, COUNT(DISTINCT column_name) as unique_columns
            FROM information_schema.columns 
            WHERE table_name = '{table}'
            GROUP BY column_name
            LIMIT {self.exploration_result_limit}
            """,
            purpose=f"Level 2: Column uniqueness analysis for {table}",
            complexity_level=2,
            execution_result=None,
            insights=[]
        ))
        
        return queries
    
    def _generate_level_3_queries(self, table: str) -> List[ExplorationQuery]:
        """Generate Level 3 queries: Basic relationships and data exploration"""
        queries = []
        
        # Pattern-based data search with fuzzy matching
        patterns = self._extract_search_patterns_from_request()
        
        for pattern in patterns[:2]:  # Limit to first 2 patterns  
            queries.append(ExplorationQuery(
                sql=f"""
                SELECT * FROM {table} 
                WHERE EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = '{table}' 
                    AND column_name ILIKE '%{pattern}%'
                )
                LIMIT {self.exploration_result_limit}
                """,
                purpose=f"Level 3: Data search for pattern '{pattern}' in {table}",
                complexity_level=3,
                execution_result=None,
                insights=[]
            ))
        
        # Foreign key relationship discovery
        queries.append(ExplorationQuery(
            sql=f"""
            SELECT 
                tc.constraint_name,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND (tc.table_name = '{table}' OR ccu.table_name = '{table}')
            LIMIT {self.exploration_result_limit}
            """,
            purpose=f"Level 3: Foreign key relationships for {table}",
            complexity_level=3,
            execution_result=None,
            insights=[]
        ))
        
        return queries
    
    def _generate_level_4_queries(self, table: str, related_tables: List[str]) -> List[ExplorationQuery]:
        """Generate Level 4 queries: Complex analysis with JOINs"""
        queries = []
        
        # Try to join with related tables (if any exist)
        for related_table in related_tables[:2]:  # Limit to 2 related tables
            if related_table == table:
                continue
                
            queries.append(ExplorationQuery(
                sql=f"""
                SELECT t1.*, t2.*
                FROM {table} t1
                JOIN {related_table} t2 ON (
                    -- Attempt common join patterns
                    t1.id = t2.{table}_id OR
                    t1.{related_table}_id = t2.id OR
                    t1.id = t2.id
                )
                LIMIT {min(10, self.exploration_result_limit)}
                """,
                purpose=f"Level 4: JOIN exploration between {table} and {related_table}",
                complexity_level=4,
                execution_result=None,
                insights=[]
            ))
        
        # Aggregation analysis
        queries.append(ExplorationQuery(
            sql=f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT id) as unique_ids
            FROM {table}
            WHERE id IS NOT NULL
            LIMIT {self.exploration_result_limit}
            """,
            purpose=f"Level 4: Aggregation analysis for {table}",
            complexity_level=4,
            execution_result=None,
            insights=[]
        ))
        
        return queries
    
    def _generate_level_5_queries(self, table: str) -> List[ExplorationQuery]:
        """Generate Level 5 queries: Advanced analytics and complex patterns"""
        queries = []
        
        # Advanced pattern analysis with CTEs (paper mentions CTE usage)
        patterns = self._extract_search_patterns_from_request()
        
        for pattern in patterns[:1]:  # Only 1 pattern for complex queries
            queries.append(ExplorationQuery(
                sql=f"""
                WITH pattern_analysis AS (
                    SELECT 
                        column_name,
                        data_type,
                        COUNT(*) as column_count
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    AND column_name ILIKE '%{pattern}%'
                    GROUP BY column_name, data_type
                )
                SELECT * FROM pattern_analysis
                LIMIT {self.exploration_result_limit}
                """,
                purpose=f"Level 5: CTE-based pattern analysis for '{pattern}' in {table}",
                complexity_level=5,
                execution_result=None,
                insights=[]
            ))
        
        # Complex data type analysis
        queries.append(ExplorationQuery(
            sql=f"""
            WITH column_stats AS (
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    CASE 
                        WHEN data_type LIKE '%char%' THEN 'text'
                        WHEN data_type LIKE '%int%' THEN 'numeric'
                        WHEN data_type LIKE '%date%' THEN 'temporal'
                        ELSE 'other'
                    END as category
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            )
            SELECT category, COUNT(*) as count
            FROM column_stats
            GROUP BY category
            LIMIT {self.exploration_result_limit}
            """,
            purpose=f"Level 5: Advanced column categorization for {table}",
            complexity_level=5,
            execution_result=None,
            insights=[]
        ))
        
        return queries
    
    async def _find_relevant_tables_with_candidates(self, tables: List[str]) -> List[str]:
        """Identify tables based on all candidates and FK relationships"""
        try:
            # Analyze all candidates to get referenced entities
            candidate_analysis = self._analyze_all_candidates()
            
            # Start with tables from all candidates
            relevant_tables = set(candidate_analysis['tables'])
            
            # Add tables based on search patterns and business terms
            pattern_tables = self._find_tables_by_patterns(tables, candidate_analysis['patterns'])
            relevant_tables.update(pattern_tables)
            
            # Expand with FK-related tables
            expanded_tables = self._expand_with_foreign_key_tables(list(relevant_tables), tables)
            relevant_tables.update(expanded_tables)
            
            # If still no relevant tables found, use LLM as fallback
            if not relevant_tables:
                logger.warning("No tables found from candidates, falling back to LLM analysis")
                return await self._find_relevant_tables_with_llm_fallback(tables)
            
            result = list(relevant_tables)
            logger.info(f"Candidate-based analysis identified {len(result)} relevant tables: {result[:10]}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get relevant tables from candidates: {e}")
            # Fallback to LLM analysis
            return await self._find_relevant_tables_with_llm_fallback(tables)
    
    def _find_tables_by_patterns(self, tables: List[str], patterns: List[str]) -> List[str]:
        """Find tables that match search patterns from the request"""
        matching_tables = set()
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # Direct matches
            for table in tables:
                table_lower = table.lower()
                
                # Exact match
                if pattern_lower == table_lower:
                    matching_tables.add(table)
                    continue
                
                # Pattern is substring of table name
                if pattern_lower in table_lower:
                    matching_tables.add(table)
                    continue
                
                # Table name is substring of pattern (for compound terms)
                if table_lower in pattern_lower:
                    matching_tables.add(table)
                    continue
                
                # Handle plural/singular variations
                if pattern_lower + 's' == table_lower or pattern_lower == table_lower + 's':
                    matching_tables.add(table)
                    continue
                
                # Handle common substitutions
                substitutions = {
                    'project': ['workunit', 'container', 'annotation'],
                    'order': ['workunit', 'container', 'sample'],
                    'coach': ['user', 'person', 'contact', 'employee'],
                    'user': ['person', 'contact', 'employee', 'login'],
                    'customer': ['client', 'user', 'contact'],
                    'service': ['application', 'instrument', 'tool'],
                    'sample': ['container', 'workunit', 'specimen']
                }
                
                if pattern_lower in substitutions:
                    for substitute in substitutions[pattern_lower]:
                        if substitute in table_lower:
                            matching_tables.add(table)
                            break
        
        logger.info(f"Pattern matching found {len(matching_tables)} tables for patterns {patterns}: {list(matching_tables)[:5]}")
        return list(matching_tables)
    
    def _expand_with_foreign_key_tables(self, base_tables: List[str], all_tables: List[str]) -> List[str]:
        """Expand table list with foreign key related tables"""
        expanded = set()
        
        for base_table in base_tables:
            # Get schema for this table to find foreign keys
            try:
                schema = self.db_manager.get_table_schema(base_table)
                fk_relationships = self._detect_foreign_keys(base_table, schema)
                
                # Add referenced tables
                for column, ref_table in fk_relationships.items():
                    if ref_table in all_tables:
                        expanded.add(ref_table)
                        logger.debug(f"Added FK-related table {ref_table} (from {base_table}.{column})")
                
            except Exception as e:
                logger.debug(f"Could not analyze FK relationships for {base_table}: {e}")
                continue
        
        # Also find tables that reference our base tables
        for table in all_tables[:50]:  # Limit to avoid performance issues
            try:
                schema = self.db_manager.get_table_schema(table)
                fk_relationships = self._detect_foreign_keys(table, schema)
                
                # Check if this table references any of our base tables
                for column, ref_table in fk_relationships.items():
                    if ref_table in base_tables:
                        expanded.add(table)
                        logger.debug(f"Added referencing table {table} (references {ref_table})")
                        break
                        
            except Exception as e:
                logger.debug(f"Could not analyze FK relationships for {table}: {e}")
                continue
        
        logger.info(f"FK expansion added {len(expanded)} related tables")
        return list(expanded)
    
    async def _find_relevant_tables_with_llm_fallback(self, tables: List[str]) -> List[str]:
        """Fallback LLM-based table selection when candidate analysis fails"""
        try:
            # Use first 100 tables for broader LLM analysis
            table_list = ', '.join(tables[:100])
            
            relevance_prompt = f"""
            Given the following user request and list of database tables, identify the 5-10 most relevant tables that would be needed to answer the request.
            
            User Request: {self.user_request}
            
            Available Tables: {table_list}
            
            Analyze which tables are most likely to contain the data needed for this request. Consider:
            - Direct matches (e.g., "project" request → "projects" table)
            - Related entities (e.g., "coach" might be in "users" or "contacts" table)
            - Foreign key relationships that would be needed
            - Supporting data for complete answers
            
            Return ONLY a comma-separated list of the most relevant table names, ordered by relevance (most relevant first).
            Do not include explanations, just the table names.
            
            Example format: table1, table2, table3, table4, table5
            """
            
            response = await self.llm_client.generate_completion(
                prompt=relevance_prompt,
                system_prompt="You are a database analyst expert at identifying relevant tables for SQL queries.",
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse the response to get table names
            if response and response.content:
                relevant_table_names = [name.strip() for name in response.content.split(',')]
                # Filter to only valid table names
                valid_tables = [table for table in relevant_table_names if table in tables]
                
                if valid_tables:
                    logger.info(f"LLM fallback identified {len(valid_tables)} relevant tables: {valid_tables}")
                    return valid_tables
                else:
                    logger.warning("LLM fallback didn't return valid table names, using first 5 tables")
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
        
        # Final fallback to first 5 tables
        return tables[:5]

    async def _find_relevant_tables_with_llm(self, tables: List[str]) -> List[str]:
        """Use LLM to identify tables most relevant to the user request"""
        try:
            # Create a prompt for table relevance analysis
            table_list = ', '.join(tables[:50])  # Limit for prompt size
            
            relevance_prompt = f"""
            Given the following user request and list of database tables, identify the 5-10 most relevant tables that would be needed to answer the request.
            
            User Request: {self.user_request}
            
            Available Tables: {table_list}
            
            Analyze which tables are most likely to contain the data needed for this request. Consider:
            - Direct matches (e.g., "project" request → "projects" table)
            - Related entities (e.g., "coach" might be in "users" or "contacts" table)
            - Foreign key relationships that would be needed
            - Supporting data for complete answers
            
            Return ONLY a comma-separated list of the most relevant table names, ordered by relevance (most relevant first).
            Do not include explanations, just the table names.
            
            Example format: table1, table2, table3, table4, table5
            """
            
            response = await self.llm_client.generate_completion(
                prompt=relevance_prompt,
                system_prompt="You are a database analyst expert at identifying relevant tables for SQL queries.",
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse the response to get table names
            if response and response.content:
                relevant_table_names = [name.strip() for name in response.content.split(',')]
                # Filter to only valid table names
                valid_tables = [table for table in relevant_table_names if table in tables]
                
                if valid_tables:
                    logger.info(f"LLM identified {len(valid_tables)} relevant tables: {valid_tables}")
                    return valid_tables
                else:
                    logger.warning("LLM didn't return valid table names, falling back to first 5 tables")
            
        except Exception as e:
            logger.error(f"Failed to get relevant tables from LLM: {e}")
        
        # Fallback to first 5 tables if LLM analysis fails
        return tables[:5]
    
    def _assess_query_complexity(self, sql: str) -> int:
        """Assess query complexity level (1-5)"""
        sql_upper = sql.upper()
        complexity = 1
        
        if 'JOIN' in sql_upper:
            complexity += 1
        if 'GROUP BY' in sql_upper or 'HAVING' in sql_upper:
            complexity += 1
        if 'DISTINCT' in sql_upper:
            complexity += 1
        if sql_upper.count('SELECT') > 1:  # Subqueries
            complexity += 1
        
        return min(5, complexity)
    
    async def _execute_exploration_queries(self, queries: List[ExplorationQuery]) -> List[ColumnInsight]:
        """Execute exploration queries progressively from simple to complex (ReFoRCE paper methodology)"""
        logger.info(f"Executing {len(queries)} exploration queries with progressive complexity...")
        
        # Sort by complexity (simple first) - ReFoRCE progression: 1-5 scale
        sorted_queries = sorted(queries, key=lambda q: q.complexity_level)
        
        # Group queries by complexity level for progressive execution
        complexity_groups = {}
        for query in sorted_queries:
            level = query.complexity_level
            if level not in complexity_groups:
                complexity_groups[level] = []
            complexity_groups[level].append(query)
        
        insights = []
        total_executed = 0
        
        # Execute queries progressively by complexity level
        for level in sorted([1, 2, 3, 4, 5]):
            if level not in complexity_groups:
                continue
                
            level_queries = complexity_groups[level]
            logger.info(f"Executing complexity level {level} queries: {len(level_queries)} queries")
            
            level_insights = []
            for i, query in enumerate(level_queries):
                try:
                    total_executed += 1
                    logger.info(f"Executing L{level} query {i+1}/{len(level_queries)} (overall {total_executed}/{len(sorted_queries)}): {query.purpose}")
                    
                    # Execute query with execution feedback
                    result = await self.sql_executor.execute_sql_safe(query.sql, timeout=10)
                    query.execution_result = result
                    
                    if result.success and result.data:
                        # Extract insights from results
                        query_insights = self._extract_insights_from_result(query, result)
                        level_insights.extend(query_insights)
                        query.insights = [insight.column_name for insight in query_insights]
                        
                        # Store in exploration history
                        self.exploration_queries.append(query)
                        
                        logger.info(f"L{level} query successful: {len(result.data)} rows, {len(query_insights)} insights")
                    else:
                        logger.warning(f"L{level} query failed: {result.error}")
                        # Store failed query for analysis
                        self.exploration_queries.append(query)
                    
                    # Add delay between queries to avoid overwhelming database
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"L{level} exploration query {i+1} failed: {e}")
                    continue
            
            # Analyze level results and decide if higher complexity is needed
            insights.extend(level_insights)
            
            # Early termination if we have sufficient insights (paper guidance)
            if len(insights) >= 20 and level >= 2:  # Sufficient insights from simple queries
                logger.info(f"Sufficient insights ({len(insights)}) gathered at complexity level {level}, stopping progression")
                break
            
            # If level failed completely, log but continue
            if not level_insights and level <= 2:
                logger.warning(f"No insights from complexity level {level}, but continuing progression")
            elif not level_insights and level > 2:
                logger.warning(f"No insights from complexity level {level}, considering stopping progression")
                break
        
        return insights
    
    def _extract_insights_from_result(self, query: ExplorationQuery, result: ExecutionResult) -> List[ColumnInsight]:
        """Extract column insights from query results"""
        insights = []
        
        if not result.data:
            return insights
        
        try:
            # Get first row to understand structure
            first_row = result.data[0]
            
            for column_name, value in first_row.items():
                # Try to infer table name from query
                table_name = self._infer_table_name(query.sql, column_name)
                
                # Gather sample values
                sample_values = [row.get(column_name) for row in result.data[:5]]
                
                # Count nulls and uniques
                null_count = sum(1 for row in result.data if row.get(column_name) is None)
                unique_values = set(row.get(column_name) for row in result.data if row.get(column_name) is not None)
                
                # Determine data type
                data_type = self._infer_data_type(sample_values)
                
                insight = ColumnInsight(
                    table_name=table_name,
                    column_name=column_name,
                    data_type=data_type,
                    sample_values=sample_values,
                    null_count=null_count,
                    unique_count=len(unique_values),
                    relationships=[],
                    usage_patterns=[]
                )
                
                # Store insight
                key = f"{table_name}.{column_name}"
                self.column_insights[key] = insight
                insights.append(insight)
                
        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")
        
        return insights
    
    def _infer_table_name(self, sql: str, column_name: str) -> str:
        """Infer table name from SQL query"""
        # Simple extraction - look for FROM clause
        import re
        
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            return from_match.group(1)
        
        return "unknown_table"
    
    def _infer_data_type(self, sample_values: List[Any]) -> str:
        """Infer data type from sample values"""
        non_null_values = [v for v in sample_values if v is not None]
        
        if not non_null_values:
            return "unknown"
        
        first_value = non_null_values[0]
        
        if isinstance(first_value, int):
            return "integer"
        elif isinstance(first_value, float):
            return "numeric"
        elif isinstance(first_value, str):
            # Check if it looks like a date
            import re
            if re.match(r'\d{4}-\d{2}-\d{2}', str(first_value)):
                return "date"
            return "text"
        else:
            return "mixed"
    
    async def _generate_improved_sql_from_insights(self, insights: List[ColumnInsight]) -> str:
        """Generate fresh SQL using exploration insights (ReFoRCE paper methodology)"""
        try:
            # Format exploration insights for enhanced prompt
            exploration_insights = self._format_exploration_insights_for_enhanced_prompt(insights)
            
            # Determine expected answer format
            answer_format = self._determine_answer_format()
            
            # Get original schema for context
            original_schema = self._get_original_schema_context()
            
            # Generate fresh SQL using exploration-enhanced prompt (not refinement)
            response = await self.llm_client.generate_completion(
                prompt=PromptTemplates.EXPLORATION_ENHANCED_GENERATION_PROMPT.format(
                    schema_text=original_schema,
                    exploration_insights=exploration_insights,
                    user_request=self.user_request,
                    answer_format=answer_format
                ),
                system_prompt=PromptTemplates.GENERATION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=8000  # Generous for Qwen thinking + complex SQL
            )
            
            # Extract SQL from response
            improved_sql = self._extract_sql_from_response(response.content)
            
            if improved_sql:
                # Apply self-refinement process with execution feedback (ReFoRCE paper)
                refined_sql = await self._apply_self_refinement_with_feedback(improved_sql)
                self.improved_candidates.append(refined_sql)
                logger.info("Generated and refined fresh SQL using exploration insights (paper methodology)")
                return refined_sql
            
            return improved_sql
            
        except Exception as e:
            logger.error(f"Failed to generate exploration-enhanced SQL: {e}")
            return ""
    
    def _format_exploration_insights_for_enhanced_prompt(self, insights: List[ColumnInsight]) -> str:
        """Format exploration insights according to ReFoRCE paper methodology"""
        if not insights:
            return "No column insights discovered during exploration."
        
        formatted_sections = [
            "=== DATABASE EXPLORATION INSIGHTS ===",
            "",
            f"Total insights discovered: {len(insights)}",
            f"Tables explored: {len(set(insight.table_name for insight in insights))}",
            f"Exploration queries executed: {len(self.exploration_queries)}",
            ""
        ]
        
        # Group insights by table for better organization
        table_insights = {}
        for insight in insights:
            table = insight.table_name
            if table not in table_insights:
                table_insights[table] = []
            table_insights[table].append(insight)
        
        # Format insights by table
        for table_name, table_insights_list in table_insights.items():
            formatted_sections.append(f"TABLE: {table_name}")
            formatted_sections.append(f"Columns discovered: {len(table_insights_list)}")
            formatted_sections.append("")
            
            for insight in table_insights_list[:10]:  # Limit per table
                # Format individual column insight
                sample_values_str = ', '.join(str(v) for v in insight.sample_values[:5] if v is not None)
                if not sample_values_str:
                    sample_values_str = "No sample data"
                
                formatted_sections.append(f"  • {insight.column_name}:")
                formatted_sections.append(f"    - Data Type: {insight.data_type}")
                formatted_sections.append(f"    - Sample Values: {sample_values_str}")
                formatted_sections.append(f"    - Unique Count: {insight.unique_count}")
                formatted_sections.append(f"    - Null Count: {insight.null_count}")
                
                if insight.relationships:
                    formatted_sections.append(f"    - Relationships: {', '.join(insight.relationships)}")
                
                if insight.usage_patterns:
                    formatted_sections.append(f"    - Patterns: {', '.join(insight.usage_patterns)}")
                
                formatted_sections.append("")
            
            formatted_sections.append("---")
            formatted_sections.append("")
        
        # Add query execution summary
        if self.exploration_queries:
            formatted_sections.append("=== EXPLORATION QUERY RESULTS ===")
            formatted_sections.append("")
            
            for query in self.exploration_queries[-5:]:  # Last 5 queries
                success_status = "✓" if query.execution_result and query.execution_result.success else "✗"
                row_count = len(query.execution_result.data) if query.execution_result and query.execution_result.data else 0
                
                formatted_sections.append(f"{success_status} {query.purpose}")
                formatted_sections.append(f"   Complexity Level: {query.complexity_level}, Rows: {row_count}")
                
                if query.insights:
                    formatted_sections.append(f"   Insights: {', '.join(query.insights[:3])}")
                
                formatted_sections.append("")
        
        return "\n".join(formatted_sections)
    
    def _determine_answer_format(self) -> str:
        """Determine expected answer format based on request analysis"""
        if not self.user_request:
            return "SQL query result set"
        
        request_lower = self.user_request.lower()
        
        # Analyze request type
        if any(word in request_lower for word in ['count', 'how many', 'number of']):
            return "Single numeric count value"
        elif any(word in request_lower for word in ['list', 'show', 'get', 'find']):
            return "Table with relevant columns and rows"
        elif any(word in request_lower for word in ['total', 'sum', 'average', 'max', 'min']):
            return "Aggregated numeric result"
        elif 'everything' in request_lower or 'all' in request_lower:
            return "Comprehensive data with all relevant attributes"
        else:
            return "Structured query result set"
    
    def _get_original_schema_context(self) -> str:
        """Get original schema context for enhanced prompt"""
        try:
            # Get the compressed schema that was used in previous stages
            from ..core.schema_compressor import SchemaCompressor
            compressor = SchemaCompressor(self.db_manager)
            return compressor.get_compressed_schema_text()
        except Exception as e:
            logger.error(f"Failed to get original schema context: {e}")
            return "Original schema information unavailable"
    
    async def _apply_self_refinement_with_feedback(self, sql: str, max_iterations: int = 5) -> str:
        """
        Apply self-refinement process with execution feedback (ReFoRCE paper methodology)
        Continues until self-consistency achieved or max iterations reached
        """
        current_sql = sql
        previous_results = []
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"Self-refinement iteration {iteration + 1}/{max_iterations}")
                
                # Execute current SQL
                execution_result = await self.sql_executor.execute_sql_safe(current_sql)
                
                if execution_result.success:
                    # Check for self-consistency (same result as previous iteration)
                    if iteration > 0 and self._check_self_consistency(execution_result, previous_results):
                        logger.info(f"Self-consistency achieved at iteration {iteration + 1}")
                        break
                    
                    # Store result for consistency check
                    previous_results.append({
                        'iteration': iteration,
                        'sql': current_sql,
                        'row_count': len(execution_result.data) if execution_result.data else 0,
                        'success': True
                    })
                    
                    # If successful and we have reasonable results, we're done
                    if execution_result.data and len(execution_result.data) > 0:
                        logger.info(f"Successful execution with {len(execution_result.data)} rows")
                        break
                
                else:
                    # Failed execution - apply refinement based on feedback
                    logger.warning(f"Execution failed: {execution_result.error}")
                    
                    refined_sql = await self._refine_sql_with_execution_feedback(
                        current_sql, 
                        execution_result.error
                    )
                    
                    if refined_sql == current_sql:
                        logger.warning("No refinement suggested, trying CTE-based fallback")
                        # Apply CTE-based refinement as fallback (paper methodology)
                        cte_sql = await self._apply_cte_based_refinement(current_sql)
                        if cte_sql != current_sql:
                            current_sql = cte_sql
                        else:
                            logger.warning("CTE fallback also failed, stopping iterations")
                            break
                    else:
                        current_sql = refined_sql
                    
                    # Store failed attempt
                    previous_results.append({
                        'iteration': iteration,
                        'sql': current_sql,
                        'error': execution_result.error,
                        'success': False
                    })
                
            except Exception as e:
                logger.error(f"Self-refinement iteration {iteration + 1} failed: {e}")
                break
        
        return current_sql
    
    def _check_self_consistency(self, current_result: Any, previous_results: List[Dict]) -> bool:
        """Check if current result is consistent with previous iteration (paper termination condition)"""
        if not previous_results:
            return False
        
        # Get the last successful result
        last_result = previous_results[-1]
        
        if not last_result.get('success', False):
            return False
        
        # Check if row counts match (simple consistency check)
        current_row_count = len(current_result.data) if current_result.data else 0
        last_row_count = last_result.get('row_count', 0)
        
        if current_row_count == last_row_count and current_row_count > 0:
            logger.info(f"Self-consistency detected: {current_row_count} rows in consecutive iterations")
            return True
        
        return False
    
    async def _refine_sql_with_execution_feedback(self, sql: str, error: str) -> str:
        """Refine SQL based on execution feedback"""
        try:
            # Use exploration insights in refinement context
            exploration_context = self._get_exploration_context_for_refinement()
            
            refinement_prompt = f"""
            Refine the following SQL query based on execution feedback and exploration insights:
            
            Original Request: {self.user_request}
            Current SQL: {sql}
            Execution Error: {error}
            
            Database Exploration Context:
            {exploration_context}
            
            Please fix the SQL query considering:
            1. The specific error encountered
            2. The exploration insights about actual table/column names
            3. Correct data types and relationships discovered
            4. Sample data patterns found during exploration
            
            Return the corrected SQL query:
            """
            
            response = await self.llm_client.generate_completion(
                prompt=refinement_prompt,
                system_prompt=PromptTemplates.REFINEMENT_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=6000
            )
            
            refined_sql = self._extract_sql_from_response(response.content)
            return refined_sql if refined_sql else sql
            
        except Exception as e:
            logger.error(f"SQL refinement failed: {e}")
            return sql
    
    def _get_exploration_context_for_refinement(self) -> str:
        """Get concise exploration context for refinement"""
        if not self.column_insights:
            return "No exploration insights available"
        
        # Provide key insights in concise format
        context_parts = []
        
        # Tables and columns discovered
        tables = set(insight.table_name for insight in self.column_insights.values())
        context_parts.append(f"Tables explored: {', '.join(list(tables)[:5])}")
        
        # Key column information
        key_columns = []
        for insight in list(self.column_insights.values())[:10]:
            key_columns.append(f"{insight.table_name}.{insight.column_name} ({insight.data_type})")
        
        if key_columns:
            context_parts.append(f"Key columns: {', '.join(key_columns)}")
        
        # Successful exploration queries
        successful_queries = [q for q in self.exploration_queries if q.execution_result and q.execution_result.success]
        if successful_queries:
            context_parts.append(f"Successful exploration queries: {len(successful_queries)}")
        
        return "; ".join(context_parts)
    
    async def _apply_cte_based_refinement(self, sql: str) -> str:
        """
        Apply CTE-based refinement to break down complex queries (ReFoRCE paper fallback)
        Converts complex queries into step-by-step CTEs for better error localization
        """
        try:
            logger.info("Applying CTE-based refinement as fallback")
            
            cte_prompt = f"""
            Break down the following complex SQL query into step-by-step Common Table Expressions (CTEs) to help localize and correct errors:
            
            Original Request: {self.user_request}
            Complex SQL: {sql}
            
            Exploration Context: {self._get_exploration_context_for_refinement()}
            
            Please rewrite the query using CTEs to:
            1. Break complex operations into smaller, testable steps
            2. Make each step explicit and debuggable
            3. Use discovered table and column names from exploration
            4. Ensure each CTE has a clear purpose
            
            Format example:
            WITH step1 AS (
                -- First operation
                SELECT ...
            ),
            step2 AS (
                -- Second operation using step1
                SELECT ... FROM step1 ...
            )
            SELECT ... FROM step2 ...
            
            CTE-based SQL Query:
            """
            
            response = await self.llm_client.generate_completion(
                prompt=cte_prompt,
                system_prompt="You are an expert at breaking down complex SQL queries into manageable CTEs for debugging.",
                temperature=0.1,
                max_tokens=8000
            )
            
            cte_sql = self._extract_sql_from_response(response.content)
            
            if cte_sql and cte_sql != sql:
                logger.info("Successfully generated CTE-based refinement")
                return cte_sql
            else:
                logger.warning("CTE-based refinement did not produce different SQL")
                return sql
                
        except Exception as e:
            logger.error(f"CTE-based refinement failed: {e}")
            return sql
    
    def _format_insights_for_llm(self, insights: List[ColumnInsight]) -> str:
        """Format insights for LLM consumption"""
        formatted_insights = []
        
        for insight in insights[:10]:  # Limit to top 10 insights
            formatted_insights.append(f"""
Table: {insight.table_name}
Column: {insight.column_name}
Type: {insight.data_type}
Sample Values: {', '.join(str(v) for v in insight.sample_values[:3] if v is not None)}
Unique Count: {insight.unique_count}
Null Count: {insight.null_count}
""")
        
        return "\n".join(formatted_insights)
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from LLM response"""
        import re
        
        # Look for SQL code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Look for SELECT statements
        select_match = re.search(r'(SELECT\s+.*?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        return response.strip()
    
    async def _analyze_uncertainty_areas(self, request: str) -> str:
        """Analyze and report on uncertainty areas"""
        if not self.uncertain_areas:
            await self._identify_uncertain_areas()
        
        result = f"""Uncertainty Analysis:
        
Original Request: {self.user_request}
Identified Uncertain Areas: {len(self.uncertain_areas)}

Areas Requiring Exploration:
"""
        
        for i, area in enumerate(self.uncertain_areas, 1):
            result += f"{i}. {area}\n"
        
        result += f"""
Recommended Exploration Strategy:
1. Execute progressive queries from simple to complex
2. Focus on data structure and relationships
3. Validate assumptions about column contents
4. Identify potential data quality issues
"""
        
        return result
    
    async def _get_column_insights(self) -> str:
        """Get detailed column insights discovered through exploration"""
        if not self.column_insights:
            return "No column insights available. Please run exploration first."
        
        result = f"""Column Insights Summary:
        
Total Insights: {len(self.column_insights)}
Exploration Queries: {len(self.exploration_queries)}

Detailed Insights:
"""
        
        for key, insight in list(self.column_insights.items())[:20]:  # Limit output
            result += f"""
{insight.table_name}.{insight.column_name}:
  • Type: {insight.data_type}
  • Sample: {', '.join(str(v) for v in insight.sample_values[:3] if v is not None)}
  • Unique Values: {insight.unique_count}
  • Null Values: {insight.null_count}
  • Relationships: {', '.join(insight.relationships) if insight.relationships else 'None discovered'}
"""
        
        return result
    
    def _extract_user_request(self, message: str) -> str:
        """Extract user request from message"""
        lines = message.split('\n')
        for line in lines:
            if 'request:' in line.lower():
                return line.split(':', 1)[1].strip()
        return message
    
    async def _handle_general_exploration_request(self, request: str) -> str:
        """Handle general exploration requests"""
        return f"""I'm the Column Exploration Agent. I help resolve low-confidence SQL generation cases through database exploration.

I can help with:
1. Explore database columns for ambiguous requests (explore_columns)
2. Analyze uncertainty areas (analyze_uncertainty)
3. Generate improved SQL based on insights (generate_improved_sql)
4. Provide column insights summary (get_column_insights)

Your request: {request}

Please provide:
- Original SQL request
- Areas of uncertainty (optional - I can identify them)
- Low-confidence indication from VotingAgent

I'll perform progressive exploration from simple to complex queries to resolve ambiguities."""
    
    def _analyze_all_candidates(self) -> Dict[str, List[str]]:
        """
        Analyze all SQL candidates to extract referenced tables and columns
        Returns dict with 'tables' and 'columns' lists for targeted exploration
        """
        referenced_entities = {
            'tables': set(),
            'columns': set(),
            'patterns': set()
        }
        
        for candidate in self.all_candidates:
            sql = candidate.get('sql', '')
            error = candidate.get('error', '')
            
            # Extract table references from SQL
            tables = self._extract_table_references(sql)
            referenced_entities['tables'].update(tables)
            
            # Extract column references from SQL
            columns = self._extract_column_references(sql)
            referenced_entities['columns'].update(columns)
            
            # Extract missing entities from error messages
            missing_entities = self._extract_missing_entities_from_error(error)
            referenced_entities['tables'].update(missing_entities.get('tables', []))
            referenced_entities['columns'].update(missing_entities.get('columns', []))
            
            # Extract search patterns from original request
            patterns = self._extract_search_patterns_from_request()
            referenced_entities['patterns'].update(patterns)
        
        # Convert sets to lists for easier handling
        result = {
            'tables': list(referenced_entities['tables']),
            'columns': list(referenced_entities['columns']),  
            'patterns': list(referenced_entities['patterns'])
        }
        
        logger.info(f"Extracted from all candidates: {len(result['tables'])} tables, {len(result['columns'])} columns, {len(result['patterns'])} patterns")
        return result
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table names referenced in SQL query"""
        import re
        
        tables = set()
        if not sql:
            return list(tables)
        
        # Remove comments and normalize
        sql_clean = re.sub(r'--.*?\n', ' ', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)
        sql_upper = sql_clean.upper()
        
        # Extract FROM clauses
        from_matches = re.findall(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_upper)
        tables.update(from_matches)
        
        # Extract JOIN clauses  
        join_matches = re.findall(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_upper)
        tables.update(join_matches)
        
        # Extract UPDATE/INSERT/DELETE table references
        update_matches = re.findall(r'(?:UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_upper)
        tables.update(update_matches)
        
        return [t.lower() for t in tables if t]
    
    def _extract_column_references(self, sql: str) -> List[str]:
        """Extract column names referenced in SQL query"""
        import re
        
        columns = set()
        if not sql:
            return list(columns)
        
        # Remove comments
        sql_clean = re.sub(r'--.*?\n', ' ', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)
        
        # Extract SELECT columns (simple patterns)
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_matches = re.findall(select_pattern, sql_clean, re.IGNORECASE | re.DOTALL)
        
        for match in select_matches:
            # Split by comma and extract column names
            columns_str = match.replace('\n', ' ').strip()
            if columns_str != '*':
                # Extract individual column references
                col_parts = re.split(r',', columns_str)
                for part in col_parts:
                    # Extract column name (handle table.column format)
                    col_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?$', part.strip(), re.IGNORECASE)
                    if col_match:
                        columns.add(col_match.group(1).lower())
        
        # Extract WHERE clause columns
        where_matches = re.findall(r'WHERE\s+.*?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]', sql_clean, re.IGNORECASE)
        columns.update([c.lower() for c in where_matches])
        
        return list(columns)
    
    def _extract_missing_entities_from_error(self, error: str) -> Dict[str, List[str]]:
        """Extract missing table/column names from database error messages"""
        import re
        
        entities = {'tables': [], 'columns': []}
        if not error:
            return entities
        
        error_lower = error.lower()
        
        # Common PostgreSQL error patterns
        table_not_exist_patterns = [
            r'relation "([^"]+)" does not exist',
            r'table "([^"]+)" does not exist',
            r'no such table: ([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        column_not_exist_patterns = [
            r'column "([^"]+)" does not exist', 
            r'no such column: ([a-zA-Z_][a-zA-Z0-9_]*)',
            r'unknown column \'([^\']+)\''
        ]
        
        # Extract missing tables
        for pattern in table_not_exist_patterns:
            matches = re.findall(pattern, error_lower)
            entities['tables'].extend(matches)
        
        # Extract missing columns
        for pattern in column_not_exist_patterns:
            matches = re.findall(pattern, error_lower)
            entities['columns'].extend(matches)
        
        return entities
    
    def _extract_search_patterns_from_request(self) -> List[str]:
        """Extract search patterns from the original user request for fuzzy matching"""
        import re
        
        patterns = set()
        if not self.user_request:
            return list(patterns)
        
        request_lower = self.user_request.lower()
        
        # Extract quoted strings
        quoted_patterns = re.findall(r'"([^"]+)"', request_lower)
        patterns.update(quoted_patterns)
        
        # Extract numbers (potential IDs)
        numbers = re.findall(r'\b\d+\b', request_lower)
        patterns.update(numbers)
        
        # Extract business terms (basic pattern)
        business_terms = re.findall(r'\b(project|order|coach|user|customer|sample|service|application|instrument|workunit|container)\b', request_lower)
        patterns.update(business_terms)
        
        return list(patterns)
    
    def get_exploration_results(self) -> Dict[str, Any]:
        """Get exploration results for other agents"""
        return {
            "column_insights": {
                key: {
                    "table": insight.table_name,
                    "column": insight.column_name,
                    "data_type": insight.data_type,
                    "sample_values": insight.sample_values,
                    "unique_count": insight.unique_count,
                    "null_count": insight.null_count
                }
                for key, insight in self.column_insights.items()
            },
            "improved_candidates": self.improved_candidates,
            "exploration_queries_count": len(self.exploration_queries),
            "insights_count": len(self.column_insights)
        }