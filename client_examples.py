#!/usr/bin/env python3
"""
Example clients for ReFoRCE OpenAI-compatible API
"""
import asyncio
import json
import requests
from openai import OpenAI
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8080"  # Your ReFoRCE API server
OPENAI_API_BASE = f"{API_BASE_URL}/v1"

def example_requests_client():
    """Example using requests library"""
    print("=== Using requests library ===")
    
    # Chat completions request
    payload = {
        "model": "reforce-text-to-sql",
        "messages": [
            {
                "role": "system",
                "content": "You are a Text-to-SQL assistant. Convert natural language to SQL queries."
            },
            {
                "role": "user", 
                "content": "Show all users who registered in the last 30 days"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print("SQL Query:", result["choices"][0]["message"]["content"])
            print("Usage:", result["usage"])
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def example_openai_client():
    """Example using OpenAI Python client"""
    print("\n=== Using OpenAI Python client ===")
    
    # Initialize client pointing to ReFoRCE API
    client = OpenAI(
        base_url=OPENAI_API_BASE,
        api_key="not-needed"  # ReFoRCE doesn't require API key
    )
    
    try:
        # Chat completion
        response = client.chat.completions.create(
            model="reforce-text-to-sql",
            messages=[
                {
                    "role": "system",
                    "content": "Return results in JSON format with sql and metadata fields."
                },
                {
                    "role": "user",
                    "content": "Find the top 10 customers by total order value"
                }
            ],
            temperature=0.1
        )
        
        print("✅ Success!")
        print("Response:", response.choices[0].message.content)
        print("Model:", response.model)
        print("Usage:", response.usage)
        
    except Exception as e:
        print(f"❌ OpenAI client failed: {e}")

def example_direct_api():
    """Example using direct ReFoRCE API endpoints"""
    print("\n=== Using direct ReFoRCE API ===")
    
    try:
        # Direct SQL generation
        response = requests.get(
            f"{API_BASE_URL}/v1/sql/direct",
            params={"query": "Count total number of products in each category"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print("Original Query:", result["query"])
            print("Generated SQL:", result["sql"])
            print("Confidence:", f"{result['confidence']:.1%}")
            print("Pipeline Stage:", result["pipeline_stage"])
            print("Processing Time:", f"{result['processing_time']:.2f}s")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Direct API failed: {e}")

def example_schema_info():
    """Example getting schema information"""
    print("\n=== Schema Information ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/v1/schema/info", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Schema Info:")
            print(f"Total Tables: {result['total_tables']}")
            print("Sample Tables:", result["tables"][:5])
            print("Largest Tables:", result["largest_tables"][:3])
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Schema info failed: {e}")

def example_health_check():
    """Example health check"""
    print("\n=== Health Check ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health Status:", result["status"])
            print("Components:")
            for component, healthy in result["components"].items():
                status = "✅" if healthy else "❌"
                print(f"  {status} {component}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

async def example_batch_requests():
    """Example batch processing multiple requests"""
    print("\n=== Batch Processing ===")
    
    queries = [
        "Show all active users",
        "Count orders by status", 
        "Find products with low inventory",
        "Get monthly sales totals",
        "List customers with no orders"
    ]
    
    client = OpenAI(
        base_url=OPENAI_API_BASE,
        api_key="not-needed"
    )
    
    try:
        tasks = []
        for query in queries:
            task = client.chat.completions.create(
                model="reforce-text-to-sql",
                messages=[
                    {"role": "system", "content": "Return just the SQL query."},
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )
            tasks.append(task)
        
        # Note: OpenAI client is synchronous, so we'll process sequentially
        results = []
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query}")
            response = client.chat.completions.create(
                model="reforce-text-to-sql",
                messages=[
                    {"role": "system", "content": "Return just the SQL query."},
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )
            results.append({
                "query": query,
                "sql": response.choices[0].message.content,
                "tokens": response.usage.total_tokens
            })
        
        print("✅ Batch Results:")
        for result in results:
            print(f"Query: {result['query']}")
            print(f"SQL: {result['sql'][:100]}...")
            print(f"Tokens: {result['tokens']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

def example_integration_with_langchain():
    """Example integration with LangChain"""
    print("\n=== LangChain Integration Example ===")
    
    try:
        from langchain.llms import OpenAI as LangChainOpenAI
        from langchain.schema import HumanMessage
        
        # Initialize LangChain with ReFoRCE endpoint
        llm = LangChainOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key="not-needed",
            model_name="reforce-text-to-sql"
        )
        
        # Use with LangChain
        query = "Generate SQL to find average order value by customer segment"
        response = llm.predict(query)
        
        print("✅ LangChain Success!")
        print("Response:", response)
        
    except ImportError:
        print("⚠️  LangChain not installed. Install with: pip install langchain")
    except Exception as e:
        print(f"❌ LangChain integration failed: {e}")

if __name__ == "__main__":
    print("🚀 ReFoRCE API Client Examples")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ ReFoRCE API server is not running!")
            print(f"Please start the server with: python api_server.py")
            exit(1)
    except:
        print("❌ Cannot connect to ReFoRCE API server!")
        print(f"Please ensure the server is running at {API_BASE_URL}")
        exit(1)
    
    # Run examples
    example_health_check()
    example_schema_info()
    example_requests_client()
    example_openai_client()
    example_direct_api()
    
    # Async example
    asyncio.run(example_batch_requests())
    
    # Optional LangChain example
    example_integration_with_langchain()
    
    print("\n✅ All examples completed!")
    print("\nIntegration options:")
    print("1. Use as drop-in replacement for OpenAI API")
    print("2. Integrate with existing LLM frameworks")
    print("3. Use direct endpoints for custom applications")
    print("4. Deploy with Docker for production use")