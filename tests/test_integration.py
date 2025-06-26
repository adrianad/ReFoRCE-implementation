"""
Integration tests for ReFoRCE Text-to-SQL system
"""
import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch

from reforce.workflows.reforce_workflow import ReFoRCEWorkflow, ReFoRCEResult
from reforce.core.database_manager import DatabaseManager
from reforce.core.schema_compressor import SchemaCompressor
from reforce.models.llm_client import LLMClient, LLMResponse
from reforce.agents.compression_agent import CompressionAgent
from reforce.agents.generation_agent import GenerationAgent
from reforce.agents.voting_agent import VotingAgent
from reforce.agents.exploration_agent import ExplorationAgent

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestReFoRCEIntegration:
    """Integration tests for the complete ReFoRCE pipeline"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing"""
        mock_db = Mock(spec=DatabaseManager)
        
        # Mock table structure
        mock_db.get_all_tables.return_value = [
            'users', 'orders', 'products', 'order_items',
            'user_sessions_20240101', 'user_sessions_20240102'
        ]
        
        # Mock schema information
        mock_db.get_table_schema.return_value = [
            {'column_name': 'id', 'data_type': 'integer', 'is_nullable': 'NO'},
            {'column_name': 'name', 'data_type': 'varchar', 'is_nullable': 'YES'},
            {'column_name': 'email', 'data_type': 'varchar', 'is_nullable': 'NO'}
        ]
        
        # Mock DDL generation
        mock_db.get_table_ddl.return_value = """
        CREATE TABLE users (
            id integer NOT NULL,
            name varchar(100),
            email varchar(255) NOT NULL
        );
        """
        
        # Mock table sizes
        mock_db.get_table_sizes.return_value = {
            'users': 1024,
            'orders': 2048,
            'products': 512
        }
        
        # Mock validation
        mock_db.validate_sql_syntax.return_value = (True, "Valid SQL")
        
        return mock_db
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing"""
        mock_llm = Mock(spec=LLMClient)
        
        # Mock completion responses
        mock_response = LLMResponse(
            content="SELECT * FROM users WHERE id = 1",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            model="test-model",
            finish_reason="stop"
        )
        
        mock_llm.generate_completion = AsyncMock(return_value=mock_response)
        mock_llm.generate_multiple_completions = AsyncMock(return_value=[mock_response] * 3)
        mock_llm.health_check = AsyncMock(return_value=True)
        
        return mock_llm
    
    @pytest.fixture
    def reforce_workflow(self, mock_db_manager, mock_llm_client):
        """Create ReFoRCE workflow with mocked dependencies"""
        workflow = ReFoRCEWorkflow(
            db_manager=mock_db_manager,
            llm_client=mock_llm_client
        )
        return workflow
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, reforce_workflow):
        """Test complete pipeline execution with mocked components"""
        
        # Test request
        user_request = "Show all users who have placed orders"
        
        # Mock SQL execution results
        with patch('reforce.core.sql_executor.SQLExecutor.execute_sql_safe') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.data = [{'id': 1, 'name': 'John', 'email': 'john@test.com'}]
            mock_execute.return_value.execution_time = 0.1
            mock_execute.return_value.rows_affected = 1
            
            # Execute pipeline
            result = await reforce_workflow.process_text_to_sql_request(user_request)
            
            # Verify result structure
            assert isinstance(result, ReFoRCEResult)
            assert result.final_sql != ""
            assert 0.0 <= result.confidence <= 1.0
            assert result.pipeline_stage in ["compression", "voting", "exploration"]
            assert isinstance(result.execution_successful, bool)
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, reforce_workflow):
        """Test system health check"""
        health_status = await reforce_workflow.health_check()
        
        assert isinstance(health_status, dict)
        assert "database" in health_status
        assert "llm" in health_status
        assert "compression_agent" in health_status
        assert "generation_agent" in health_status
        assert "voting_agent" in health_status
        assert "exploration_agent" in health_status
    
    @pytest.mark.asyncio
    async def test_compression_stage(self, mock_db_manager, mock_llm_client):
        """Test database compression stage"""
        compression_agent = CompressionAgent(
            db_manager=mock_db_manager,
            llm_client=mock_llm_client
        )
        
        # Test compression
        from autogen_agentchat.messages import TextMessage
        message = TextMessage(content="compress_schema", source="test")
        
        response = await compression_agent.on_messages([message], None)
        
        assert response is not None
        assert "compression" in response.chat_message.content.lower()
    
    @pytest.mark.asyncio
    async def test_generation_stage(self, mock_db_manager, mock_llm_client):
        """Test candidate generation stage"""
        generation_agent = GenerationAgent(
            db_manager=mock_db_manager,
            llm_client=mock_llm_client
        )
        
        # Set mock schema
        generation_agent.set_compressed_schema("CREATE TABLE users (id INT, name VARCHAR(100));")
        
        # Test generation
        from autogen_agentchat.messages import TextMessage
        message = TextMessage(
            content="generate_candidates\nuser_request: Show all users",
            source="test"
        )
        
        with patch('reforce.core.sql_executor.SQLExecutor.execute_sql_safe') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.data = []
            
            response = await generation_agent.on_messages([message], None)
            
            assert response is not None
            assert "candidates" in response.chat_message.content.lower()
    
    @pytest.mark.asyncio
    async def test_voting_stage(self, mock_llm_client):
        """Test majority voting stage"""
        voting_agent = VotingAgent(llm_client=mock_llm_client)
        
        # Set mock candidates
        candidates = [
            {"sql": "SELECT * FROM users", "confidence": 0.8, "success": True},
            {"sql": "SELECT id, name FROM users", "confidence": 0.7, "success": True},
            {"sql": "SELECT users.* FROM users", "confidence": 0.6, "success": True}
        ]
        
        voting_agent.set_candidates_and_context(
            candidates, 
            "Show all users", 
            "CREATE TABLE users (id INT, name VARCHAR(100));"
        )
        
        # Test voting
        from autogen_agentchat.messages import TextMessage
        message = TextMessage(content="vote_candidates", source="test")
        
        response = await voting_agent.on_messages([message], None)
        
        assert response is not None
        assert "voting" in response.chat_message.content.lower()
    
    @pytest.mark.asyncio
    async def test_exploration_stage(self, mock_db_manager, mock_llm_client):
        """Test column exploration stage"""
        exploration_agent = ExplorationAgent(
            db_manager=mock_db_manager,
            llm_client=mock_llm_client
        )
        
        # Set exploration context
        exploration_agent.set_exploration_context(
            "Show complex user data",
            ["Unclear column structure", "Ambiguous relationships"]
        )
        
        # Test exploration
        from autogen_agentchat.messages import TextMessage
        message = TextMessage(content="explore_columns", source="test")
        
        with patch('reforce.core.sql_executor.SQLExecutor.execute_sql_safe') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.data = [
                {'column_name': 'id', 'data_type': 'integer'},
                {'column_name': 'name', 'data_type': 'varchar'}
            ]
            
            response = await exploration_agent.on_messages([message], None)
            
            assert response is not None
            assert "exploration" in response.chat_message.content.lower()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, reforce_workflow):
        """Test batch processing of multiple requests"""
        requests = [
            "Show all users",
            "Count total orders",
            "Find most popular products"
        ]
        
        with patch('reforce.core.sql_executor.SQLExecutor.execute_sql_safe') as mock_execute:
            mock_execute.return_value = AsyncMock()
            mock_execute.return_value.success = True
            mock_execute.return_value.data = [{'result': 'mock'}]
            
            results = await reforce_workflow.batch_process(requests)
            
            assert len(results) == len(requests)
            assert all(isinstance(result, ReFoRCEResult) for result in results)

class TestSchemaCompression:
    """Test schema compression functionality"""
    
    @pytest.fixture
    def schema_compressor(self, mock_db_manager):
        return SchemaCompressor(mock_db_manager)
    
    def test_pattern_extraction(self, schema_compressor):
        """Test table pattern extraction"""
        tables = [
            'user_sessions_20240101',
            'user_sessions_20240102', 
            'user_sessions_20240103',
            'order_log_2024_01',
            'order_log_2024_02',
            'products',
            'categories'
        ]
        
        patterns = schema_compressor.extract_table_patterns(tables)
        
        # Should identify date patterns and prefixes
        assert len(patterns) > 0
        
        # Check for expected patterns
        pattern_names = list(patterns.keys())
        assert any('user_sessions' in pattern for pattern in pattern_names)
        assert any('order_log' in pattern for pattern in pattern_names)
    
    def test_representative_selection(self, schema_compressor):
        """Test representative table selection"""
        tables = ['table_1', 'table_2', 'table_main', 'table_2024']
        
        representative = schema_compressor.select_representative_table(tables)
        
        # Should prefer non-numeric/date suffixed tables
        assert representative in tables
        assert representative != 'table_2024'  # Should avoid date suffix
    
    def test_compression_calculation(self, schema_compressor, mock_db_manager):
        """Test compression ratio calculation"""
        # Mock DDL responses for size calculation
        mock_db_manager.get_table_ddl.return_value = "CREATE TABLE test (id INT);" * 100
        
        compressed_groups, ungrouped_tables, compression_ratio = schema_compressor.compress_database_schema()
        
        assert isinstance(compression_ratio, float)
        assert 0.0 <= compression_ratio <= 1.0
        assert isinstance(compressed_groups, dict)
        assert isinstance(ungrouped_tables, list)

class TestValidation:
    """Test validation utilities"""
    
    def test_sql_security_validation(self):
        """Test SQL security validation"""
        from reforce.utils.validation import SQLValidator
        
        # Safe SQL
        safe_sql = "SELECT id, name FROM users WHERE id = 1"
        result = SQLValidator.validate_sql_security(safe_sql)
        assert result.is_valid
        assert len(result.issues) == 0
        
        # Dangerous SQL
        dangerous_sql = "DROP TABLE users; SELECT * FROM users"
        result = SQLValidator.validate_sql_security(dangerous_sql)
        assert not result.is_valid
        assert len(result.issues) > 0
    
    def test_sql_syntax_validation(self):
        """Test SQL syntax validation"""
        from reforce.utils.validation import SQLValidator
        
        # Valid SQL
        valid_sql = "SELECT id, name FROM users WHERE id = 1"
        result = SQLValidator.validate_sql_syntax(valid_sql)
        assert result.is_valid
        
        # Invalid SQL (unbalanced parentheses)
        invalid_sql = "SELECT id, name FROM users WHERE (id = 1"
        result = SQLValidator.validate_sql_syntax(invalid_sql)
        assert not result.is_valid
        assert any("parentheses" in issue.message.lower() for issue in result.issues)
    
    def test_column_reference_validation(self):
        """Test column reference validation"""
        from reforce.utils.validation import SQLValidator
        
        available_tables = {
            'users': ['id', 'name', 'email'],
            'orders': ['id', 'user_id', 'total']
        }
        
        # Valid references
        valid_sql = "SELECT users.id, users.name FROM users"
        result = SQLValidator.validate_column_references(valid_sql, available_tables)
        assert result.is_valid
        
        # Invalid column reference
        invalid_sql = "SELECT users.invalid_column FROM users"
        result = SQLValidator.validate_column_references(invalid_sql, available_tables)
        assert not result.is_valid

class TestUtilities:
    """Test utility functions"""
    
    def test_schema_utils_ddl_parsing(self):
        """Test DDL parsing utilities"""
        from reforce.utils.schema_utils import SchemaUtils
        
        ddl = """
        CREATE TABLE users (
            id integer NOT NULL,
            name varchar(100),
            email varchar(255) NOT NULL DEFAULT 'user@example.com'
        );
        """
        
        table_info = SchemaUtils.parse_ddl_statement(ddl)
        
        assert table_info.name == 'users'
        assert len(table_info.columns) == 3
        
        # Check column parsing
        id_column = next(col for col in table_info.columns if col['name'] == 'id')
        assert id_column['data_type'] == 'integer'
        assert not id_column['is_nullable']
    
    def test_schema_complexity_analysis(self):
        """Test schema complexity analysis"""
        from reforce.utils.schema_utils import SchemaUtils, TableInfo
        
        tables = [
            TableInfo(
                name='users',
                columns=[
                    {'name': 'id', 'data_type': 'integer'},
                    {'name': 'name', 'data_type': 'varchar'}
                ]
            ),
            TableInfo(
                name='orders',
                columns=[
                    {'name': 'id', 'data_type': 'integer'},
                    {'name': 'user_id', 'data_type': 'integer'},
                    {'name': 'total', 'data_type': 'decimal'}
                ]
            )
        ]
        
        analysis = SchemaUtils.analyze_schema_complexity(tables)
        
        assert 'total_tables' in analysis
        assert 'total_columns' in analysis
        assert 'complexity_score' in analysis
        assert analysis['total_tables'] == 2
        assert analysis['total_columns'] == 5

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])