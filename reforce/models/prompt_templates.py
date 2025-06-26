"""
Prompt templates for ReFoRCE Text-to-SQL system
"""

class PromptTemplates:
    """Collection of prompt templates for different stages of ReFoRCE"""
    
    # Database Information Compression Stage
    COMPRESSION_SYSTEM_PROMPT = """
    You are a database schema analysis expert. Your task is to analyze database schemas and identify tables that can be grouped together based on similar patterns, structures, or naming conventions.
    
    Focus on:
    1. Tables with similar prefixes or suffixes
    2. Tables with temporal patterns (dates, versions)
    3. Tables with similar column structures
    4. Tables that serve similar purposes
    
    Always prioritize accuracy and preserve essential schema information.
    """
    
    COMPRESSION_ANALYSIS_PROMPT = """
    Analyze the following database schema and identify table groupings:
    
    {schema_text}
    
    Please identify:
    1. Tables that can be grouped by naming patterns
    2. Tables with similar structures that serve related purposes  
    3. The most representative table for each group
    4. Essential schema information that must be preserved
    
    Provide your analysis in a structured format.
    """
    
    # Candidate Generation Stage
    GENERATION_SYSTEM_PROMPT = """
    You are an expert SQL developer specializing in PostgreSQL. Your task is to generate accurate SQL queries based on natural language requests and database schema information.
    
    Key requirements:
    1. Generate syntactically correct PostgreSQL SQL
    2. Use appropriate table and column names from the provided schema
    3. Consider query performance and best practices
    4. Handle complex queries with JOINs, subqueries, and aggregations
    5. Return only the SQL query without explanations unless requested
    """
    
    GENERATION_PROMPT = """
    Given the following database schema and natural language request, generate a PostgreSQL SQL query.
    
    Database Schema:
    {schema_text}
    
    Natural Language Request:
    {user_request}
    
    Additional Context:
    {context}
    
    Generate a precise SQL query that fulfills the request. Consider:
    - Exact table and column names from the schema
    - Proper JOIN conditions
    - Appropriate WHERE clauses and filters
    - Correct aggregation functions if needed
    - Performance considerations
    
    SQL Query:
    """
    
    # Exploration-Enhanced Generation (ReFoRCE Stage 4)
    EXPLORATION_ENHANCED_GENERATION_PROMPT = """
    Generate a PostgreSQL SQL query using the original schema and exploration insights discovered through database analysis.
    
    Original Database Schema (𝒫init):
    {schema_text}
    
    Column Exploration Data (𝒫column + ℛexploration):
    {exploration_insights}
    
    Natural Language Request:
    {user_request}
    
    Expected Answer Format (ℱ):
    {answer_format}
    
    IMPORTANT: Use the exploration insights to understand:
    - Actual column names and data types discovered
    - Foreign key relationships found through analysis
    - Sample data patterns and value ranges
    - Table relationships and join conditions
    - Data quality and null patterns
    
    The exploration insights provide real database structure information that may differ from or enhance the original schema. 
    Prioritize the discovered insights when generating the SQL query.
    
    Generate a precise SQL query that leverages both the original schema and exploration discoveries:
    """
    
    # Self-Refinement Stage
    REFINEMENT_SYSTEM_PROMPT = """
    You are a SQL query optimizer and error correction specialist. Your task is to analyze SQL queries for errors and improve them based on execution feedback.
    
    Focus on:
    1. Syntax errors and corrections
    2. Semantic errors (wrong table/column references)
    3. Logic errors in query structure
    4. Performance optimizations
    5. Best practices compliance
    """
    
    REFINEMENT_PROMPT = """
    Review and refine the following SQL query based on the execution feedback:
    
    Original Request: {original_request}
    Current SQL Query: {sql_query}
    Execution Feedback: {feedback}
    Error Details: {error_details}
    
    Database Schema Reference:
    {schema_text}
    
    Please provide an improved SQL query that addresses the issues identified. If the query is already correct, return it unchanged.
    
    Improved SQL Query:
    """
    
    # Voting and Consensus Stage
    VOTING_SYSTEM_PROMPT = """
    You are a SQL query evaluation expert. Your task is to analyze multiple SQL query candidates and determine which one best fulfills the original request.
    
    Evaluation criteria:
    1. Correctness of the query logic
    2. Proper use of database schema
    3. Efficiency and performance
    4. Completeness of the solution
    5. Adherence to best practices
    """
    
    VOTING_PROMPT = """
    Evaluate the following SQL query candidates for the given request:
    
    Original Request: {user_request}
    Database Schema: {schema_text}
    
    Candidate Queries:
    {candidate_queries}
    
    For each candidate, provide:
    1. A correctness score (0-10)
    2. Key strengths and weaknesses
    3. Your confidence level (0.0-1.0)
    
    Then select the best candidate and explain your reasoning.
    
    Evaluation:
    """
    
    # Column Exploration Stage
    EXPLORATION_SYSTEM_PROMPT = """
    You are a database exploration specialist. Your task is to generate SQL queries to explore and understand database column contents, especially for ambiguous or complex schemas.
    
    Focus on:
    1. Understanding column data types and contents
    2. Identifying relationships between tables
    3. Exploring nested or complex data structures
    4. Finding relevant data for query construction
    """
    
    EXPLORATION_PROMPT = """
    Generate exploratory SQL queries to better understand the database structure for the following request:
    
    Original Request: {user_request}
    Available Tables: {table_list}
    Uncertain Areas: {uncertainty_areas}
    
    Generate 3-5 simple exploratory queries that will help understand:
    1. Column contents and data types
    2. Data relationships
    3. Sample values
    4. Data distribution
    
    Start with simple queries and gradually increase complexity. Limit results to 100 rows or 5KB.
    
    Exploratory Queries:
    """
    
    # Format Restriction Templates
    FORMAT_RESTRICTION_PROMPT = """
    Your response must follow this exact format:
    
    ```sql
    -- SQL query here
    SELECT ...
    FROM ...
    WHERE ...
    ```
    
    Do not include any explanations or additional text unless specifically requested.
    """
    
    CSV_FORMAT_RESTRICTION = """
    Return results in CSV format with the following structure:
    - First row: column headers
    - Subsequent rows: data values
    - Use proper escaping for special characters
    - Limit to first 1000 rows if result set is large
    """
    
    # Error Analysis Templates
    ERROR_ANALYSIS_PROMPT = """
    Analyze the following SQL execution error:
    
    SQL Query: {sql_query}
    Error Message: {error_message}
    Database Schema: {schema_text}
    
    Identify:
    1. The root cause of the error
    2. Specific corrections needed
    3. Alternative approaches if applicable
    
    Provide a corrected version of the query.
    """
    
    # Schema Linking Templates
    SCHEMA_LINKING_PROMPT = """
    Given the natural language request and database schema, identify the most relevant tables and columns:
    
    Request: {user_request}
    Schema: {schema_text}
    
    Identify:
    1. Primary tables needed for the query
    2. Key columns that match the request
    3. Necessary JOIN relationships
    4. Potential ambiguities or missing information
    
    Provide a structured mapping of request elements to schema elements.
    """
    
    @classmethod
    def get_compression_prompt(cls, schema_text: str) -> str:
        """Get formatted compression analysis prompt"""
        return cls.COMPRESSION_ANALYSIS_PROMPT.format(schema_text=schema_text)
    
    @classmethod
    def get_generation_prompt(cls, schema_text: str, user_request: str, context: str = "") -> str:
        """Get formatted SQL generation prompt"""
        return cls.GENERATION_PROMPT.format(
            schema_text=schema_text,
            user_request=user_request,
            context=context
        )
    
    @classmethod
    def get_refinement_prompt(cls, original_request: str, sql_query: str, 
                            feedback: str, error_details: str, schema_text: str) -> str:
        """Get formatted refinement prompt"""
        return cls.REFINEMENT_PROMPT.format(
            original_request=original_request,
            sql_query=sql_query,
            feedback=feedback,
            error_details=error_details,
            schema_text=schema_text
        )
    
    @classmethod
    def get_voting_prompt(cls, user_request: str, schema_text: str, candidate_queries: str) -> str:
        """Get formatted voting prompt"""
        return cls.VOTING_PROMPT.format(
            user_request=user_request,
            schema_text=schema_text,
            candidate_queries=candidate_queries
        )
    
    @classmethod
    def get_exploration_prompt(cls, user_request: str, table_list: str, uncertainty_areas: str) -> str:
        """Get formatted exploration prompt"""
        return cls.EXPLORATION_PROMPT.format(
            user_request=user_request,
            table_list=table_list,
            uncertainty_areas=uncertainty_areas
        )
    
    @classmethod
    def get_error_analysis_prompt(cls, sql_query: str, error_message: str, schema_text: str) -> str:
        """Get formatted error analysis prompt"""
        return cls.ERROR_ANALYSIS_PROMPT.format(
            sql_query=sql_query,
            error_message=error_message,
            schema_text=schema_text
        )
    
    @classmethod
    def get_schema_linking_prompt(cls, user_request: str, schema_text: str) -> str:
        """Get formatted schema linking prompt"""
        return cls.SCHEMA_LINKING_PROMPT.format(
            user_request=user_request,
            schema_text=schema_text
        )