"""
Majority Voting and Consensus Agent for ReFoRCE
Stage 3: Implements voting mechanism and consensus enforcement
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Sequence
from dataclasses import dataclass
from collections import Counter
import statistics
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.base import Response, TaskResult
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from ..models.llm_client import LLMClient
from ..models.prompt_templates import PromptTemplates
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class VoteResult:
    """Result of voting process"""
    winning_candidate: str
    confidence: float
    vote_distribution: Dict[int, int]
    consensus_strength: float
    is_high_confidence: bool

@dataclass
class CandidateEvaluation:
    """Evaluation of a single candidate"""
    candidate_id: int
    sql: str
    correctness_score: float
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    vote_weight: float

class VotingAgent(AssistantAgent):
    """
    AutoGen agent for majority voting and consensus enforcement
    Implements ReFoRCE Stage 3: Majority-Vote Consensus
    """
    
    def __init__(
        self,
        name: str = "VotingAgent",
        description: str = "SQL candidate voting and consensus specialist",
        llm_client: Optional[LLMClient] = None,
        model_client: Optional[OpenAIChatCompletionClient] = None
    ):
        # Initialize components
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
        
        # Configuration
        self.config = settings.reforce
        self.confidence_threshold = self.config.confidence_threshold
        self.min_votes_required = self.config.min_votes_required
        
        # State tracking
        self.candidates = []
        self.evaluations = []
        self.vote_result = None
        self.user_request = None
        self.compressed_schema = None
    
    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """Handle incoming messages for voting and consensus"""
        try:
            latest_message = messages[-1]
            request_content = latest_message.content
            
            logger.info(f"VotingAgent received request: {request_content[:100]}...")
            
            # Parse request type
            if "vote_candidates" in request_content.lower():
                result = await self._vote_on_candidates(request_content)
            elif "evaluate_candidates" in request_content.lower():
                result = await self._evaluate_candidates(request_content)
            elif "consensus_check" in request_content.lower():
                result = await self._check_consensus()
            elif "get_winner" in request_content.lower():
                result = await self._get_voting_winner()
            else:
                result = await self._handle_general_voting_request(request_content)
            
            return Response(
                chat_message=TextMessage(content=result, source=self.name)
            )
            
        except Exception as e:
            error_msg = f"VotingAgent error: {str(e)}"
            logger.error(error_msg)
            return Response(
                chat_message=TextMessage(content=error_msg, source=self.name)
            )
    
    def set_candidates_and_context(self, candidates: List[Dict], user_request: str, schema: str):
        """Set candidates from GenerationAgent and context"""
        self.candidates = candidates
        self.user_request = user_request
        self.compressed_schema = schema
        logger.info(f"Received {len(candidates)} candidates for voting")
    
    async def _vote_on_candidates(self, request: str) -> str:
        """Perform voting process on all candidates"""
        try:
            if not self.candidates:
                return "Error: No candidates available for voting. Please provide candidates from GenerationAgent."
            
            logger.info(f"Starting voting process for {len(self.candidates)} candidates")
            
            # Step 1: Evaluate all candidates
            await self._evaluate_all_candidates()
            
            # Step 2: Conduct voting rounds
            vote_result = await self._conduct_voting_rounds()
            
            # Step 3: Check consensus
            consensus_result = await self._determine_consensus(vote_result)
            
            self.vote_result = consensus_result
            
            result = f"""Voting Process Completed!
            
User Request: {self.user_request}
Total Candidates: {len(self.candidates)}

Voting Results:
- Winner: Candidate {consensus_result.winning_candidate}
- Confidence: {consensus_result.confidence:.2f}
- Consensus Strength: {consensus_result.consensus_strength:.2f}
- High Confidence: {'Yes' if consensus_result.is_high_confidence else 'No'}

Vote Distribution:
"""
            
            for candidate_id, votes in consensus_result.vote_distribution.items():
                result += f"  Candidate {candidate_id}: {votes} votes\n"
            
            # Add recommendation
            if consensus_result.is_high_confidence:
                result += "\n✓ High confidence result - recommended for execution"
            else:
                result += "\n⚠ Low confidence result - consider Column Exploration Agent"
            
            return result
            
        except Exception as e:
            logger.error(f"Voting process failed: {e}")
            raise
    
    async def _evaluate_all_candidates(self):
        """Evaluate all candidates for voting"""
        logger.info("Evaluating all candidates...")
        
        evaluation_tasks = []
        for i, candidate in enumerate(self.candidates):
            task = self._evaluate_single_candidate(i, candidate)
            evaluation_tasks.append(task)
        
        evaluations = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        valid_evaluations = []
        for eval_result in evaluations:
            if isinstance(eval_result, Exception):
                logger.error(f"Candidate evaluation failed: {eval_result}")
            else:
                valid_evaluations.append(eval_result)
        
        self.evaluations = valid_evaluations
        logger.info(f"Completed evaluation of {len(valid_evaluations)} candidates")
    
    async def _evaluate_single_candidate(self, candidate_id: int, candidate: Dict) -> CandidateEvaluation:
        """Evaluate a single candidate"""
        try:
            # Prepare candidate info for evaluation
            candidate_sql = candidate.get('sql', '')
            
            # Format candidates for voting prompt
            candidate_text = f"Candidate {candidate_id + 1}:\n```sql\n{candidate_sql}\n```"
            
            # Get LLM evaluation
            evaluation_response = await self.llm_client.generate_completion(
                prompt=PromptTemplates.get_voting_prompt(
                    user_request=self.user_request,
                    schema_text=self.compressed_schema[:5000],  # Limit schema size
                    candidate_queries=candidate_text
                ),
                system_prompt=PromptTemplates.VOTING_SYSTEM_PROMPT,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            # Parse evaluation response
            evaluation_data = self._parse_evaluation_response(evaluation_response.content, candidate_id)
            
            return CandidateEvaluation(
                candidate_id=candidate_id,
                sql=candidate_sql,
                correctness_score=evaluation_data.get('correctness_score', 0.5),
                confidence=evaluation_data.get('confidence', 0.5),
                strengths=evaluation_data.get('strengths', []),
                weaknesses=evaluation_data.get('weaknesses', []),
                vote_weight=self._calculate_vote_weight(candidate, evaluation_data)
            )
            
        except Exception as e:
            logger.error(f"Single candidate evaluation failed: {e}")
            # Return default evaluation
            return CandidateEvaluation(
                candidate_id=candidate_id,
                sql=candidate.get('sql', ''),
                correctness_score=0.5,
                confidence=0.3,
                strengths=[],
                weaknesses=[f"Evaluation failed: {str(e)}"],
                vote_weight=0.1
            )
    
    def _parse_evaluation_response(self, response: str, candidate_id: int) -> Dict[str, Any]:
        """Parse LLM evaluation response"""
        try:
            evaluation_data = {
                'correctness_score': 0.5,
                'confidence': 0.5,
                'strengths': [],
                'weaknesses': []
            }
            
            # Try to extract scores using regex
            import re
            
            # Look for correctness score
            correctness_match = re.search(r'correctness.*?(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if correctness_match:
                score = float(correctness_match.group(1))
                if score > 10:  # Assume 10-point scale
                    score = score / 10.0
                evaluation_data['correctness_score'] = min(1.0, max(0.0, score))
            
            # Look for confidence score
            confidence_match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if confidence_match:
                conf = float(confidence_match.group(1))
                if conf > 1.0:  # Assume percentage or 10-point scale
                    conf = conf / 100.0 if conf > 10 else conf / 10.0
                evaluation_data['confidence'] = min(1.0, max(0.0, conf))
            
            # Extract strengths and weaknesses
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if 'strength' in line.lower():
                    current_section = 'strengths'
                elif 'weakness' in line.lower() or 'issue' in line.lower():
                    current_section = 'weaknesses'
                elif line.startswith('-') or line.startswith('•'):
                    if current_section == 'strengths':
                        evaluation_data['strengths'].append(line[1:].strip())
                    elif current_section == 'weaknesses':
                        evaluation_data['weaknesses'].append(line[1:].strip())
            
            return evaluation_data
            
        except Exception as e:
            logger.warning(f"Could not parse evaluation response: {e}")
            return {
                'correctness_score': 0.5,
                'confidence': 0.5,
                'strengths': [],
                'weaknesses': []
            }
    
    def _calculate_vote_weight(self, candidate: Dict, evaluation_data: Dict) -> float:
        """Calculate vote weight for a candidate"""
        weight = 0.0
        
        # Base weight from correctness score
        weight += evaluation_data.get('correctness_score', 0.5) * 0.4
        
        # Confidence weight
        weight += evaluation_data.get('confidence', 0.5) * 0.3
        
        # Execution success bonus
        if candidate.get('success', False):
            weight += 0.2
        
        # Validation score
        weight += candidate.get('validation_score', 0.5) * 0.1
        
        return min(1.0, max(0.0, weight))
    
    async def _conduct_voting_rounds(self) -> Dict[int, int]:
        """Conduct multiple voting rounds for robustness"""
        logger.info("Conducting voting rounds...")
        
        # Conduct multiple voting rounds with different approaches
        voting_rounds = []
        
        # Round 1: Weight-based voting
        round1_votes = self._weight_based_voting()
        voting_rounds.append(round1_votes)
        
        # Round 2: Pairwise comparison voting
        round2_votes = await self._pairwise_comparison_voting()
        voting_rounds.append(round2_votes)
        
        # Round 3: Holistic evaluation voting
        round3_votes = await self._holistic_evaluation_voting()
        voting_rounds.append(round3_votes)
        
        # Aggregate votes across rounds
        aggregated_votes = self._aggregate_voting_rounds(voting_rounds)
        
        return aggregated_votes
    
    def _weight_based_voting(self) -> Dict[int, int]:
        """Vote based on calculated weights"""
        votes = {}
        
        if not self.evaluations:
            return votes
        
        # Convert weights to votes (each evaluation gets 1 vote for highest weight)
        for evaluation in self.evaluations:
            votes[evaluation.candidate_id] = int(evaluation.vote_weight * 10)  # Scale to 0-10 votes
        
        return votes
    
    async def _pairwise_comparison_voting(self) -> Dict[int, int]:
        """Vote based on pairwise comparisons"""
        votes = {i: 0 for i in range(len(self.candidates))}
        
        if len(self.evaluations) < 2:
            return votes
        
        # Compare each pair of candidates
        for i in range(len(self.evaluations)):
            for j in range(i + 1, len(self.evaluations)):
                eval_i = self.evaluations[i]
                eval_j = self.evaluations[j]
                
                # Simple comparison based on combined score
                score_i = eval_i.correctness_score * eval_i.confidence
                score_j = eval_j.correctness_score * eval_j.confidence
                
                if score_i > score_j:
                    votes[eval_i.candidate_id] += 1
                elif score_j > score_i:
                    votes[eval_j.candidate_id] += 1
        
        return votes
    
    async def _holistic_evaluation_voting(self) -> Dict[int, int]:
        """Vote based on holistic LLM evaluation"""
        votes = {i: 0 for i in range(len(self.candidates))}
        
        try:
            # Prepare all candidates for comparison
            candidates_text = ""
            for i, evaluation in enumerate(self.evaluations):
                candidates_text += f"""
Candidate {i + 1}:
SQL: {evaluation.sql}
Correctness Score: {evaluation.correctness_score:.2f}
Confidence: {evaluation.confidence:.2f}
Strengths: {', '.join(evaluation.strengths[:3])}
Weaknesses: {', '.join(evaluation.weaknesses[:3])}
"""
            
            # Get holistic evaluation
            holistic_prompt = f"""
            Compare all SQL candidates for the following request and select the best one:
            
            Request: {self.user_request}
            
            Candidates:
            {candidates_text}
            
            Select the best candidate (1-{len(self.evaluations)}) and provide reasoning.
            Format: "Best candidate: X"
            """
            
            response = await self.llm_client.generate_completion(
                prompt=holistic_prompt,
                system_prompt=PromptTemplates.VOTING_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            # Parse winner
            import re
            winner_match = re.search(r'best candidate:?\s*(\d+)', response.content, re.IGNORECASE)
            if winner_match:
                winner_id = int(winner_match.group(1)) - 1  # Convert to 0-based index
                if 0 <= winner_id < len(self.evaluations):
                    votes[winner_id] = 5  # Strong vote weight
            
        except Exception as e:
            logger.error(f"Holistic evaluation voting failed: {e}")
        
        return votes
    
    def _aggregate_voting_rounds(self, voting_rounds: List[Dict[int, int]]) -> Dict[int, int]:
        """Aggregate votes from multiple rounds"""
        aggregated_votes = {}
        
        # Sum votes across all rounds
        for vote_round in voting_rounds:
            for candidate_id, votes in vote_round.items():
                aggregated_votes[candidate_id] = aggregated_votes.get(candidate_id, 0) + votes
        
        return aggregated_votes
    
    async def _determine_consensus(self, vote_distribution: Dict[int, int]) -> VoteResult:
        """Determine consensus from vote distribution"""
        if not vote_distribution:
            return VoteResult(
                winning_candidate="0",
                confidence=0.0,
                vote_distribution={},
                consensus_strength=0.0,
                is_high_confidence=False
            )
        
        # Find winner
        winner_id = max(vote_distribution.keys(), key=lambda k: vote_distribution[k])
        winner_votes = vote_distribution[winner_id]
        total_votes = sum(vote_distribution.values())
        
        # Calculate confidence
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0
        
        # Calculate consensus strength (how much the winner dominates)
        vote_values = list(vote_distribution.values())
        vote_values.sort(reverse=True)
        
        if len(vote_values) >= 2:
            consensus_strength = (vote_values[0] - vote_values[1]) / vote_values[0]
        else:
            consensus_strength = 1.0
        
        # Determine if high confidence
        is_high_confidence = (
            confidence >= self.confidence_threshold and
            winner_votes >= self.min_votes_required and
            consensus_strength >= 0.3
        )
        
        return VoteResult(
            winning_candidate=str(winner_id),
            confidence=confidence,
            vote_distribution=vote_distribution,
            consensus_strength=consensus_strength,
            is_high_confidence=is_high_confidence
        )
    
    async def _check_consensus(self) -> str:
        """Check current consensus status"""
        if not self.vote_result:
            return "No voting results available. Please run voting process first."
        
        result = f"""Consensus Analysis:
        
Winner: Candidate {self.vote_result.winning_candidate}
Confidence Level: {self.vote_result.confidence:.2%}
Consensus Strength: {self.vote_result.consensus_strength:.2%}
High Confidence: {'Yes' if self.vote_result.is_high_confidence else 'No'}

Recommendation: """
        
        if self.vote_result.is_high_confidence:
            result += "✓ Proceed with winning candidate - high confidence consensus achieved"
        else:
            result += "⚠ Low confidence - recommend Column Exploration Agent for additional analysis"
        
        return result
    
    async def _get_voting_winner(self) -> str:
        """Get the winning candidate details"""
        if not self.vote_result:
            return "No voting results available. Please run voting process first."
        
        winner_id = int(self.vote_result.winning_candidate)
        
        if winner_id < len(self.candidates):
            winner_candidate = self.candidates[winner_id]
            winner_evaluation = None
            
            # Find corresponding evaluation
            for eval_item in self.evaluations:
                if eval_item.candidate_id == winner_id:
                    winner_evaluation = eval_item
                    break
            
            result = f"""Winning SQL Candidate:
            
Candidate ID: {winner_id}
Confidence: {self.vote_result.confidence:.2%}

```sql
{winner_candidate.get('sql', 'SQL not available')}
```

Evaluation Details:
"""
            
            if winner_evaluation:
                result += f"""- Correctness Score: {winner_evaluation.correctness_score:.2f}
- Evaluation Confidence: {winner_evaluation.confidence:.2f}
- Vote Weight: {winner_evaluation.vote_weight:.2f}

Strengths:
{chr(10).join(f"  • {s}" for s in winner_evaluation.strengths)}

Weaknesses:
{chr(10).join(f"  • {w}" for w in winner_evaluation.weaknesses)}"""
            
            return result
        else:
            return f"Error: Winner ID {winner_id} is invalid"
    
    async def _handle_general_voting_request(self, request: str) -> str:
        """Handle general voting requests"""
        return f"""I'm the Majority Voting and Consensus Agent. I can help with:

1. Vote on SQL candidates (vote_candidates)
2. Evaluate candidates individually (evaluate_candidates)  
3. Check consensus strength (consensus_check)
4. Get winning candidate details (get_winner)

Your request: {request}

Please provide:
- SQL candidates from GenerationAgent
- Original user request
- Compressed database schema

Then specify which voting operation you'd like me to perform."""
    
    def get_voting_results(self) -> Dict[str, Any]:
        """Get current voting results for other agents"""
        if not self.vote_result:
            return {"status": "no_results"}
        
        return {
            "winner_candidate_id": int(self.vote_result.winning_candidate),
            "confidence": self.vote_result.confidence,
            "consensus_strength": self.vote_result.consensus_strength,
            "is_high_confidence": self.vote_result.is_high_confidence,
            "vote_distribution": self.vote_result.vote_distribution,
            "recommendation": "proceed" if self.vote_result.is_high_confidence else "explore_columns"
        }