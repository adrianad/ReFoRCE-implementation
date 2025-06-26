"""
Column Exploration Agent for ReFoRCE
Stage 4: Iterative column exploration for low-confidence cases
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response

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
        llm_client: Optional[LLMClient] = None
    ):
        super().__init__(name=name, description=description)
        
        self.db_manager = db_manager or DatabaseManager()
        self.llm_client = llm_client or LLMClient()
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
    
    async def on_messages(self, messages: List[TextMessage], cancellation_token) -> Response:
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
    
    def set_exploration_context(self, user_request: str, uncertain_areas: List[str]):
        """Set exploration context from VotingAgent"""
        self.user_request = user_request
        self.uncertain_areas = uncertain_areas
        logger.info(f"Exploration context set with {len(uncertain_areas)} uncertain areas")
    
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
        
        # Add some standard exploration queries
        queries.extend(self._generate_standard_exploration_queries(available_tables[:10]))
        
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
    
    def _generate_standard_exploration_queries(self, tables: List[str]) -> List[ExplorationQuery]:
        """Generate standard exploration queries for common patterns"""
        standard_queries = []
        
        for table in tables[:3]:  # Limit to first 3 tables
            # Basic table exploration
            standard_queries.append(ExplorationQuery(
                sql=f"SELECT * FROM {table} LIMIT 5",
                purpose=f"Sample data from {table}",
                complexity_level=1,
                execution_result=None,
                insights=[]
            ))
            
            # Column information
            standard_queries.append(ExplorationQuery(
                sql=f"""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                ORDER BY ordinal_position
                """,
                purpose=f"Column structure of {table}",
                complexity_level=1,
                execution_result=None,
                insights=[]
            ))
        
        return standard_queries
    
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
        """Execute exploration queries progressively from simple to complex"""
        logger.info(f"Executing {len(queries)} exploration queries...")
        
        # Sort by complexity (simple first)
        sorted_queries = sorted(queries, key=lambda q: q.complexity_level)
        
        insights = []
        
        for i, query in enumerate(sorted_queries):
            try:
                logger.info(f"Executing exploration query {i+1}/{len(sorted_queries)}: {query.purpose}")
                
                # Execute query
                result = await self.sql_executor.execute_sql_safe(query.sql, timeout=10)
                query.execution_result = result
                
                if result.success and result.data:
                    # Extract insights from results
                    query_insights = self._extract_insights_from_result(query, result)
                    insights.extend(query_insights)
                    query.insights = [insight.column_name for insight in query_insights]
                    
                    # Store in exploration history
                    self.exploration_queries.append(query)
                    
                    logger.info(f"Query successful: {len(result.data)} rows, {len(query_insights)} insights")
                else:
                    logger.warning(f"Query failed: {result.error}")
                
                # Add delay between queries to avoid overwhelming database
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Exploration query {i+1} failed: {e}")
                continue
        
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
        """Generate improved SQL based on exploration insights"""
        try:
            # Prepare insights summary
            insights_text = self._format_insights_for_llm(insights)
            
            # Generate improved SQL
            improvement_prompt = f"""
            Based on the following database exploration insights, generate an improved SQL query for the original request:
            
            Original Request: {self.user_request}
            
            Exploration Insights:
            {insights_text}
            
            Generate an optimized SQL query that leverages the discovered insights about column structures, data types, and relationships.
            """
            
            response = await self.llm_client.generate_completion(
                prompt=improvement_prompt,
                system_prompt=PromptTemplates.GENERATION_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            # Extract SQL from response
            improved_sql = self._extract_sql_from_response(response.content)
            
            if improved_sql:
                self.improved_candidates.append(improved_sql)
            
            return improved_sql
            
        except Exception as e:
            logger.error(f"Failed to generate improved SQL: {e}")
            return ""
    
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