"""
Database Information Compression Agent for ReFoRCE
Stage 1: Implements pattern-based table grouping and schema compression
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Sequence
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.base import Response, TaskResult
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from ..core.database_manager import DatabaseManager
from ..core.schema_compressor import SchemaCompressor
from ..models.llm_client import LLMClient
from ..models.prompt_templates import PromptTemplates
from ..config.settings import settings

logger = logging.getLogger(__name__)

class CompressionAgent(AssistantAgent):
    """
    AutoGen agent responsible for database schema compression
    Implements ReFoRCE Stage 1: Database Information Compression
    """
    
    def __init__(
        self,
        name: str = "CompressionAgent",
        description: str = "Database schema compression specialist",
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
        
        self.schema_compressor = SchemaCompressor(self.db_manager)
        
        # State tracking
        self.compressed_schema = None
        self.compression_stats = None
        self.analysis_results = None
    
    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Handle incoming messages and perform schema compression
        """
        try:
            # Get the latest message
            latest_message = messages[-1]
            request_content = latest_message.content
            
            logger.info(f"CompressionAgent received request: {request_content[:100]}...")
            
            # Parse request type
            if "compress_schema" in request_content.lower():
                result = await self._compress_database_schema()
            elif "analyze_patterns" in request_content.lower():
                result = await self._analyze_table_patterns()
            elif "get_compressed_schema" in request_content.lower():
                result = await self._get_compressed_schema_text()
            elif "compression_stats" in request_content.lower():
                result = await self._get_compression_statistics()
            else:
                result = await self._handle_general_compression_request(request_content)
            
            return Response(
                chat_message=TextMessage(content=result, source=self.name)
            )
            
        except Exception as e:
            error_msg = f"CompressionAgent error: {str(e)}"
            logger.error(error_msg)
            return Response(
                chat_message=TextMessage(content=error_msg, source=self.name)
            )
    
    async def _compress_database_schema(self) -> str:
        """
        Perform complete database schema compression
        """
        try:
            logger.info("Starting database schema compression...")
            
            # Get compression results
            compressed_groups, ungrouped_tables, compression_ratio = self.schema_compressor.compress_database_schema()
            
            # Store results
            self.compression_stats = {
                "compression_ratio": compression_ratio,
                "total_groups": len(compressed_groups),
                "ungrouped_tables": len(ungrouped_tables),
                "total_tables": sum(len(group.tables) for group in compressed_groups.values()) + len(ungrouped_tables)
            }
            
            # Generate compressed schema text
            self.compressed_schema = self.schema_compressor.get_compressed_schema_text()
            
            # Use LLM to analyze and enhance compression
            await self._llm_analyze_compression(compressed_groups)
            
            result = f"""Database schema compression completed successfully!
            
Compression Statistics:
- Compression Ratio: {compression_ratio:.2%}
- Total Tables: {self.compression_stats['total_tables']}
- Compressed Groups: {self.compression_stats['total_groups']}
- Ungrouped Tables: {self.compression_stats['ungrouped_tables']}

The compressed schema is ready for use in subsequent ReFoRCE stages.
Schema size reduced significantly while preserving essential information."""
            
            logger.info("Schema compression completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Schema compression failed: {e}")
            raise
    
    async def _analyze_table_patterns(self) -> str:
        """
        Analyze table naming patterns and grouping opportunities
        """
        try:
            # Get all tables
            all_tables = self.db_manager.get_all_tables()
            
            # Extract patterns
            patterns = self.schema_compressor.extract_table_patterns(all_tables)
            
            # Use LLM for additional pattern analysis
            tables_text = "\n".join(all_tables)
            llm_response = await self.llm_client.generate_completion(
                prompt=PromptTemplates.get_compression_prompt(tables_text),
                system_prompt=PromptTemplates.COMPRESSION_SYSTEM_PROMPT
            )
            
            self.analysis_results = {
                "total_tables": len(all_tables),
                "identified_patterns": len(patterns),
                "patterns": patterns,
                "llm_analysis": llm_response.content
            }
            
            result = f"""Table Pattern Analysis Results:
            
Total Tables: {len(all_tables)}
Identified Patterns: {len(patterns)}

Pattern Groups:
"""
            
            for pattern, tables in patterns.items():
                result += f"\n- {pattern}: {len(tables)} tables"
                result += f"\n  Examples: {', '.join(tables[:3])}{'...' if len(tables) > 3 else ''}"
            
            result += f"\n\nLLM Analysis:\n{llm_response.content}"
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            raise
    
    async def _get_compressed_schema_text(self) -> str:
        """
        Return the compressed schema text for use by other agents
        """
        if not self.compressed_schema:
            # Generate compressed schema if not already done
            await self._compress_database_schema()
        
        return f"""Compressed Database Schema:

{self.compressed_schema}

Note: This schema has been compressed using ReFoRCE's pattern-based approach.
Total compression ratio: {self.compression_stats.get('compression_ratio', 0.0):.2%}"""
    
    async def _get_compression_statistics(self) -> str:
        """
        Return detailed compression statistics
        """
        if not self.compression_stats:
            await self._compress_database_schema()
        
        # Get detailed statistics
        summary = self.schema_compressor.get_table_groups_summary()
        
        result = f"""Detailed Compression Statistics:

Overall Statistics:
- Compression Ratio: {summary['compression_ratio']:.2%}
- Total Groups: {summary['total_groups']}
- Ungrouped Tables: {summary['total_ungrouped']}

Group Details:
"""
        
        for pattern, group_info in summary['groups'].items():
            result += f"""
- Pattern: {pattern}
  - Tables: {group_info['table_count']}
  - Representative: {group_info['representative']}
  - Group Compression: {group_info['compression_ratio']:.2%}
  - Original Size: {group_info['original_size']:,} bytes
  - Compressed Size: {group_info['compressed_size']:,} bytes
"""
        
        return result
    
    async def _handle_general_compression_request(self, request: str) -> str:
        """
        Handle general compression-related requests
        """
        try:
            # Use LLM to understand the request and determine appropriate action
            analysis_prompt = f"""
            You are the Database Information Compression Agent in the ReFoRCE system.
            A user has made the following request: {request}
            
            Based on this request, determine if they want:
            1. Full schema compression
            2. Pattern analysis only  
            3. Compression statistics
            4. Access to compressed schema
            5. Something else related to database compression
            
            Respond with the most appropriate action and any additional context needed.
            """
            
            llm_response = await self.llm_client.generate_completion(
                prompt=analysis_prompt,
                system_prompt=PromptTemplates.COMPRESSION_SYSTEM_PROMPT
            )
            
            # Based on LLM analysis, route to appropriate method
            analysis = llm_response.content.lower()
            
            if "full schema compression" in analysis or "compress" in analysis:
                return await self._compress_database_schema()
            elif "pattern analysis" in analysis:
                return await self._analyze_table_patterns()
            elif "statistics" in analysis:
                return await self._get_compression_statistics()
            elif "compressed schema" in analysis:
                return await self._get_compressed_schema_text()
            else:
                return f"""I'm the Database Information Compression Agent. I can help with:

1. Compress database schema (reduces size by ~96%)
2. Analyze table naming patterns
3. Provide compression statistics
4. Generate compressed schema text for other agents

Your request: {request}

LLM Analysis: {llm_response.content}

Please specify what compression task you'd like me to perform."""
            
        except Exception as e:
            logger.error(f"General request handling failed: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    async def _llm_analyze_compression(self, compressed_groups: Dict) -> None:
        """
        Use LLM to analyze and validate compression results
        """
        try:
            # Prepare summary for LLM analysis
            groups_summary = []
            for pattern, group in compressed_groups.items():
                groups_summary.append(f"Pattern: {pattern} ({len(group.tables)} tables)")
            
            analysis_prompt = f"""
            Analyze the following database schema compression results:
            
            Compression Groups:
            {chr(10).join(groups_summary)}
            
            Evaluate:
            1. Quality of pattern identification
            2. Appropriateness of table groupings
            3. Potential improvements
            4. Risks of information loss
            
            Provide recommendations for optimization.
            """
            
            response = await self.llm_client.generate_completion(
                prompt=analysis_prompt,
                system_prompt=PromptTemplates.COMPRESSION_SYSTEM_PROMPT
            )
            
            # Store LLM analysis
            if not hasattr(self, 'llm_compression_analysis'):
                self.llm_compression_analysis = {}
            
            self.llm_compression_analysis['validation'] = response.content
            logger.info("LLM compression analysis completed")
            
        except Exception as e:
            logger.warning(f"LLM compression analysis failed: {e}")
    
    def get_compression_results(self) -> Dict[str, Any]:
        """
        Get current compression results for other agents
        """
        return {
            "compressed_schema": self.compressed_schema,
            "compression_stats": self.compression_stats,
            "analysis_results": self.analysis_results,
            "llm_analysis": getattr(self, 'llm_compression_analysis', None)
        }
    
    def is_ready(self) -> bool:
        """
        Check if compression has been completed and results are available
        """
        return (
            self.compressed_schema is not None and 
            self.compression_stats is not None
        )