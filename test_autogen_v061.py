#!/usr/bin/env python3
"""
Test script for AutoGen v0.6.1 integration with ReFoRCE
"""
import asyncio
import sys
import logging
from unittest.mock import Mock, AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_autogen_imports():
    """Test AutoGen v0.6.1 imports"""
    print("🧪 Testing AutoGen v0.6.1 imports...")
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import ChatMessage, TextMessage
        from autogen_agentchat.base import Response, TaskResult
        from autogen_core import CancellationToken
        
        print("✅ All AutoGen v0.6.1 imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ AutoGen import failed: {e}")
        return False

async def test_compression_agent():
    """Test compression agent with v0.6.1 API"""
    print("\n🧪 Testing CompressionAgent with v0.6.1...")
    
    try:
        # Mock dependencies
        mock_db = Mock()
        mock_db.get_all_tables.return_value = ["users", "orders", "products"]
        mock_db.get_table_schema.return_value = []
        mock_db.get_table_ddl.return_value = "CREATE TABLE test (id INT);"
        
        mock_llm = Mock()
        mock_llm.generate_completion = AsyncMock()
        mock_llm.generate_completion.return_value = Mock(content="Test response")
        
        from reforce.agents.compression_agent import CompressionAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        
        agent = CompressionAgent(
            db_manager=mock_db,
            llm_client=mock_llm
        )
        
        # Test with v0.6.1 message format
        message = TextMessage(content="compress_schema", source="test")
        cancellation_token = CancellationToken()
        
        # Test the on_messages method
        response = await agent.on_messages([message], cancellation_token)
        
        print("✅ CompressionAgent works with v0.6.1 API")
        return True
        
    except Exception as e:
        print(f"❌ CompressionAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_generation_agent():
    """Test generation agent with v0.6.1 API"""
    print("\n🧪 Testing GenerationAgent with v0.6.1...")
    
    try:
        # Mock dependencies
        mock_db = Mock()
        mock_llm = Mock()
        mock_llm.generate_multiple_completions = AsyncMock()
        mock_llm.generate_multiple_completions.return_value = [
            Mock(content="SELECT * FROM users", usage={}, model="test", finish_reason="stop")
        ]
        
        from reforce.agents.generation_agent import GenerationAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        
        agent = GenerationAgent(
            db_manager=mock_db,
            llm_client=mock_llm
        )
        
        # Test with v0.6.1 message format
        message = TextMessage(content="generate_candidates\nuser_request: Show all users", source="test")
        cancellation_token = CancellationToken()
        
        response = await agent.on_messages([message], cancellation_token)
        
        print("✅ GenerationAgent works with v0.6.1 API")
        return True
        
    except Exception as e:
        print(f"❌ GenerationAgent test failed: {e}")
        return False

async def test_voting_agent():
    """Test voting agent with v0.6.1 API"""
    print("\n🧪 Testing VotingAgent with v0.6.1...")
    
    try:
        mock_llm = Mock()
        mock_llm.generate_completion = AsyncMock()
        mock_llm.generate_completion.return_value = Mock(content="Voting analysis complete")
        
        from reforce.agents.voting_agent import VotingAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        
        agent = VotingAgent(llm_client=mock_llm)
        
        # Set some test candidates
        agent.candidates = [
            {"sql": "SELECT * FROM users", "confidence": 0.8},
            {"sql": "SELECT id, name FROM users", "confidence": 0.7}
        ]
        
        message = TextMessage(content="vote_candidates", source="test")
        cancellation_token = CancellationToken()
        
        response = await agent.on_messages([message], cancellation_token)
        
        print("✅ VotingAgent works with v0.6.1 API")
        return True
        
    except Exception as e:
        print(f"❌ VotingAgent test failed: {e}")
        return False

async def test_exploration_agent():
    """Test exploration agent with v0.6.1 API"""
    print("\n🧪 Testing ExplorationAgent with v0.6.1...")
    
    try:
        mock_db = Mock()
        mock_llm = Mock()
        mock_llm.generate_completion = AsyncMock()
        mock_llm.generate_completion.return_value = Mock(content="Exploration complete")
        
        from reforce.agents.exploration_agent import ExplorationAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        
        agent = ExplorationAgent(
            db_manager=mock_db,
            llm_client=mock_llm
        )
        
        message = TextMessage(content="explore_columns", source="test")
        cancellation_token = CancellationToken()
        
        response = await agent.on_messages([message], cancellation_token)
        
        print("✅ ExplorationAgent works with v0.6.1 API")
        return True
        
    except Exception as e:
        print(f"❌ ExplorationAgent test failed: {e}")
        return False

async def main():
    """Run all v0.6.1 compatibility tests"""
    print("🚀 ReFoRCE AutoGen v0.6.1 Compatibility Test")
    print("=" * 50)
    
    test_results = {}
    
    # Test AutoGen imports
    test_results["autogen_imports"] = await test_autogen_imports()
    
    # Test all agents
    test_results["compression_agent"] = await test_compression_agent()
    test_results["generation_agent"] = await test_generation_agent()
    test_results["voting_agent"] = await test_voting_agent()
    test_results["exploration_agent"] = await test_exploration_agent()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = passed / total * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate == 100:
        print("🎉 Perfect! ReFoRCE is fully compatible with AutoGen v0.6.1")
        print("💡 Ready to use: python main.py --diagnostics")
    elif success_rate >= 80:
        print("✅ Good compatibility with AutoGen v0.6.1")
        print("💡 Minor issues detected - check specific test failures")
    else:
        print("❌ Compatibility issues detected")
        print("💡 Consider updating package versions or checking imports")
    
    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)