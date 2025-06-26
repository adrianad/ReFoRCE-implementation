"""
ReFoRCE AutoGen Workflow Orchestration
Coordinates all four stages of the ReFoRCE Text-to-SQL pipeline
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage

from ..agents.compression_agent import CompressionAgent
from ..agents.generation_agent import GenerationAgent
from ..agents.voting_agent import VotingAgent
from ..agents.exploration_agent import ExplorationAgent
from ..core.database_manager import DatabaseManager
from ..models.llm_client import LLMClient
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ReFoRCEResult:
    """Final result from ReFoRCE pipeline"""
    final_sql: str
    confidence: float
    pipeline_stage: str
    execution_successful: bool
    compression_ratio: float
    candidates_generated: int
    exploration_performed: bool
    processing_time: float
    metadata: Dict[str, Any]

class ReFoRCEWorkflow:
    """
    Main workflow orchestrator for ReFoRCE Text-to-SQL pipeline
    Implements the complete 4-stage process using AutoGen
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.db_manager = db_manager or DatabaseManager()
        self.llm_client = llm_client or LLMClient()
        
        # Initialize agents
        self.compression_agent = CompressionAgent(
            db_manager=self.db_manager,
            llm_client=self.llm_client
        )
        self.generation_agent = GenerationAgent(
            db_manager=self.db_manager,
            llm_client=self.llm_client
        )
        self.voting_agent = VotingAgent(
            llm_client=self.llm_client
        )
        self.exploration_agent = ExplorationAgent(
            db_manager=self.db_manager,
            llm_client=self.llm_client
        )
        
        # Workflow state
        self.current_request = None
        self.pipeline_results = {}
        self.final_result = None
    
    async def process_text_to_sql_request(self, user_request: str) -> ReFoRCEResult:
        """
        Process a complete text-to-SQL request through the ReFoRCE pipeline
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting ReFoRCE pipeline for request: {user_request[:100]}...")
            self.current_request = user_request
            
            # Stage 1: Database Information Compression
            logger.info("=== Stage 1: Database Information Compression ===")
            compression_result = await self._execute_compression_stage()
            
            # Stage 2: Candidate Generation with Self-Refinement
            logger.info("=== Stage 2: Candidate Generation & Self-Refinement ===")
            generation_result = await self._execute_generation_stage(compression_result, user_request)
            
            # Stage 3: Majority Voting and Consensus Enforcement
            logger.info("=== Stage 3: Majority Voting & Consensus ===")
            voting_result = await self._execute_voting_stage(generation_result, user_request, compression_result)
            
            # Stage 4: Column Exploration (if needed)
            exploration_result = None
            if not voting_result.get('is_high_confidence', False):
                logger.info("=== Stage 4: Column Exploration (Low Confidence) ===")
                exploration_result = await self._execute_exploration_stage(user_request, voting_result)
            
            # Compile final result
            processing_time = time.time() - start_time
            final_result = self._compile_final_result(
                compression_result, generation_result, voting_result, exploration_result, processing_time
            )
            
            self.final_result = final_result
            logger.info(f"ReFoRCE pipeline completed in {processing_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"ReFoRCE pipeline failed: {e}")
            raise
    
    async def _execute_compression_stage(self) -> Dict[str, Any]:
        """Execute Stage 1: Database Information Compression"""
        try:
            # Create compression request message
            compression_message = TextMessage(
                content="compress_schema",
                source="ReFoRCEWorkflow"
            )
            
            # Process compression
            response = await self.compression_agent.on_messages([compression_message], None)
            
            # Get compression results
            compression_results = self.compression_agent.get_compression_results()
            
            logger.info(f"Compression completed: {compression_results.get('compression_stats', {}).get('compression_ratio', 0):.2%} reduction")
            
            return compression_results
            
        except Exception as e:
            logger.error(f"Compression stage failed: {e}")
            raise
    
    async def _execute_generation_stage(self, compression_result: Dict, user_request: str) -> Dict[str, Any]:
        """Execute Stage 2: Candidate Generation with Self-Refinement"""
        try:
            # Set compressed schema for generation agent
            compressed_schema = compression_result.get('compressed_schema', '')
            self.generation_agent.set_compressed_schema(compressed_schema)
            
            # Create generation request
            generation_message = TextMessage(
                content=f"generate_candidates\nuser_request: {user_request}",
                source="ReFoRCEWorkflow"
            )
            
            # Process generation
            response = await self.generation_agent.on_messages([generation_message], None)
            
            # Get generation results
            generation_results = self.generation_agent.get_generation_results()
            
            logger.info(f"Generation completed: {generation_results.get('total_candidates', 0)} candidates generated")
            
            return generation_results
            
        except Exception as e:
            logger.error(f"Generation stage failed: {e}")
            raise
    
    async def _execute_voting_stage(self, generation_result: Dict, user_request: str, compression_result: Dict) -> Dict[str, Any]:
        """Execute Stage 3: Majority Voting and Consensus Enforcement"""
        try:
            # Set candidates and context for voting agent
            candidates = generation_result.get('candidates', [])
            compressed_schema = compression_result.get('compressed_schema', '')
            
            self.voting_agent.set_candidates_and_context(candidates, user_request, compressed_schema)
            
            # Create voting request
            voting_message = TextMessage(
                content="vote_candidates",
                source="ReFoRCEWorkflow"
            )
            
            # Process voting
            response = await self.voting_agent.on_messages([voting_message], None)
            
            # Get voting results
            voting_results = self.voting_agent.get_voting_results()
            
            logger.info(f"Voting completed: Confidence {voting_results.get('confidence', 0):.2f}, High confidence: {voting_results.get('is_high_confidence', False)}")
            
            return voting_results
            
        except Exception as e:
            logger.error(f"Voting stage failed: {e}")
            raise
    
    async def _execute_exploration_stage(self, user_request: str, voting_result: Dict) -> Dict[str, Any]:
        """Execute Stage 4: Column Exploration (for low confidence cases)"""
        try:
            # Set exploration context
            uncertain_areas = ["Low confidence from voting stage"]
            self.exploration_agent.set_exploration_context(user_request, uncertain_areas)
            
            # Create exploration request
            exploration_message = TextMessage(
                content="explore_columns",
                source="ReFoRCEWorkflow"
            )
            
            # Process exploration
            response = await self.exploration_agent.on_messages([exploration_message], None)
            
            # Get exploration results
            exploration_results = self.exploration_agent.get_exploration_results()
            
            logger.info(f"Exploration completed: {exploration_results.get('insights_count', 0)} insights discovered")
            
            return exploration_results
            
        except Exception as e:
            logger.error(f"Exploration stage failed: {e}")
            # Return empty result rather than failing entire pipeline
            return {"exploration_performed": False, "error": str(e)}
    
    def _compile_final_result(
        self,
        compression_result: Dict,
        generation_result: Dict,
        voting_result: Dict,
        exploration_result: Optional[Dict],
        processing_time: float
    ) -> ReFoRCEResult:
        """Compile final result from all pipeline stages"""
        
        # Determine final SQL
        final_sql = ""
        confidence = 0.0
        pipeline_stage = "unknown"
        
        if exploration_result and exploration_result.get('improved_candidates'):
            # Use improved SQL from exploration
            final_sql = exploration_result['improved_candidates'][0]
            confidence = 0.8  # Boost confidence after exploration
            pipeline_stage = "exploration"
        elif voting_result.get('winner_candidate_id') is not None:
            # Use winner from voting
            winner_id = voting_result['winner_candidate_id']
            candidates = generation_result.get('candidates', [])
            if winner_id < len(candidates):
                final_sql = candidates[winner_id].get('sql', '')
                confidence = voting_result.get('confidence', 0.0)
                pipeline_stage = "voting"
        
        # Test execution
        execution_successful = False
        if final_sql:
            try:
                # Quick syntax validation
                from ..core.sql_executor import SQLExecutor
                sql_executor = SQLExecutor(self.db_manager)
                validation = sql_executor.validate_sql_syntax(final_sql)
                execution_successful = validation.is_valid
            except:
                execution_successful = False
        
        return ReFoRCEResult(
            final_sql=final_sql,
            confidence=confidence,
            pipeline_stage=pipeline_stage,
            execution_successful=execution_successful,
            compression_ratio=compression_result.get('compression_stats', {}).get('compression_ratio', 0.0),
            candidates_generated=generation_result.get('total_candidates', 0),
            exploration_performed=exploration_result is not None,
            processing_time=processing_time,
            metadata={
                "compression": compression_result,
                "generation": generation_result,
                "voting": voting_result,
                "exploration": exploration_result
            }
        )
    
    async def interactive_session(self):
        """Run an interactive ReFoRCE session"""
        console = Console()
        
        print("=== ReFoRCE Text-to-SQL Interactive Session ===")
        print("Enter your natural language SQL requests. Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                user_request = input("SQL Request: ").strip()
                
                if user_request.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_request:
                    continue
                
                print(f"\nProcessing: {user_request}")
                print("=" * 60)
                
                # Process request through pipeline
                result = await self.process_text_to_sql_request(user_request)
                
                # Display results
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error processing request: {e}")
                logger.error(f"Interactive session error: {e}")
    
    def _display_result(self, result: ReFoRCEResult):
        """Display formatted result to user"""
        print(f"\n🎯 Final SQL ({result.pipeline_stage.upper()} stage):")
        print("=" * 60)
        print(f"```sql\n{result.final_sql}\n```")
        
        print(f"\n📊 Pipeline Statistics:")
        print(f"• Confidence: {result.confidence:.2%}")
        print(f"• Processing Time: {result.processing_time:.2f}s")
        print(f"• Compression Ratio: {result.compression_ratio:.2%}")
        print(f"• Candidates Generated: {result.candidates_generated}")
        print(f"• Exploration Performed: {'Yes' if result.exploration_performed else 'No'}")
        print(f"• Execution Ready: {'✓' if result.execution_successful else '✗'}")
        
        if result.confidence < 0.7:
            print(f"\n⚠️  Low confidence result - manual review recommended")
        elif result.exploration_performed:
            print(f"\n✨ Enhanced through column exploration")
        else:
            print(f"\n✅ High confidence result")
        
        print("\n" + "=" * 60)
    
    async def batch_process(self, requests: List[str]) -> List[ReFoRCEResult]:
        """Process multiple requests in batch"""
        results = []
        
        for i, request in enumerate(requests, 1):
            try:
                logger.info(f"Processing batch request {i}/{len(requests)}")
                result = await self.process_text_to_sql_request(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch request {i} failed: {e}")
                # Add error result
                results.append(ReFoRCEResult(
                    final_sql="",
                    confidence=0.0,
                    pipeline_stage="error",
                    execution_successful=False,
                    compression_ratio=0.0,
                    candidates_generated=0,
                    exploration_performed=False,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        if not self.final_result:
            return {"status": "no_results"}
        
        stats = {
            "pipeline_completed": True,
            "final_confidence": self.final_result.confidence,
            "processing_time": self.final_result.processing_time,
            "stages_completed": 3 + (1 if self.final_result.exploration_performed else 0),
            "compression_effectiveness": self.final_result.compression_ratio,
            "generation_diversity": self.final_result.candidates_generated,
            "execution_ready": self.final_result.execution_successful
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components"""
        health_status = {}
        
        try:
            # Check database connection
            health_status["database"] = True
            self.db_manager.get_all_tables()  # Simple connectivity test
        except:
            health_status["database"] = False
        
        try:
            # Check LLM service
            health_status["llm"] = await self.llm_client.health_check()
        except:
            health_status["llm"] = False
        
        # Check agents initialization
        health_status["compression_agent"] = hasattr(self.compression_agent, 'db_manager')
        health_status["generation_agent"] = hasattr(self.generation_agent, 'llm_client')
        health_status["voting_agent"] = hasattr(self.voting_agent, 'llm_client')
        health_status["exploration_agent"] = hasattr(self.exploration_agent, 'db_manager')
        
        return health_status