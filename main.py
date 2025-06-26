#!/usr/bin/env python3
"""
ReFoRCE Text-to-SQL System - Main CLI Interface
"""
import asyncio
import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from reforce.workflows.reforce_workflow import ReFoRCEWorkflow, ReFoRCEResult
from reforce.config.settings import settings
from reforce.utils.validation import ComprehensiveValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reforce.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ReFoRCECLI:
    """Command-line interface for ReFoRCE Text-to-SQL system"""
    
    def __init__(self):
        self.workflow = None
    
    async def initialize(self):
        """Initialize the ReFoRCE workflow"""
        try:
            logger.info("Initializing ReFoRCE Text-to-SQL system...")
            self.workflow = ReFoRCEWorkflow()
            
            # Perform health check
            health_status = await self.workflow.health_check()
            
            # Report health status
            for component, status in health_status.items():
                status_icon = "✓" if status else "✗"
                logger.info(f"{status_icon} {component}: {'healthy' if status else 'unhealthy'}")
            
            if not all(health_status.values()):
                logger.warning("Some components are unhealthy - functionality may be limited")
            
            logger.info("ReFoRCE initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReFoRCE: {e}")
            raise
    
    async def process_single_request(self, request: str, output_format: str = "text") -> Dict[str, Any]:
        """Process a single text-to-SQL request"""
        try:
            logger.info(f"Processing request: {request[:100]}...")
            
            # Process through ReFoRCE pipeline
            result = await self.workflow.process_text_to_sql_request(request)
            
            # Format output
            if output_format.lower() == "json":
                return self._format_json_output(result)
            else:
                self._display_text_output(result)
                return {"status": "success"}
                
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            if output_format.lower() == "json":
                return {"status": "error", "error": str(e)}
            else:
                print(f"❌ Error: {e}")
                return {"status": "error"}
    
    async def process_batch_file(self, file_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Process requests from a batch file"""
        try:
            # Read requests from file
            with open(file_path, 'r') as f:
                requests = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Processing {len(requests)} requests from {file_path}")
            
            # Process batch
            results = await self.workflow.batch_process(requests)
            
            # Prepare output
            batch_results = {
                "total_requests": len(requests),
                "successful": sum(1 for r in results if r.execution_successful),
                "results": [self._result_to_dict(r) for r in results]
            }
            
            # Save to output file if specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(batch_results, f, indent=2)
                logger.info(f"Batch results saved to {output_file}")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def interactive_mode(self):
        """Run interactive mode"""
        print("🚀 ReFoRCE Text-to-SQL Interactive Mode")
        print("=" * 50)
        print("Enter your natural language SQL requests.")
        print("Commands: 'help', 'status', 'settings', 'quit'")
        print("=" * 50)
        
        try:
            await self.workflow.interactive_session()
        except KeyboardInterrupt:
            print("\nInteractive session terminated.")
    
    def _format_json_output(self, result: ReFoRCEResult) -> Dict[str, Any]:
        """Format result as JSON"""
        return {
            "status": "success",
            "sql": result.final_sql,
            "confidence": result.confidence,
            "pipeline_stage": result.pipeline_stage,
            "execution_ready": result.execution_successful,
            "statistics": {
                "compression_ratio": result.compression_ratio,
                "candidates_generated": result.candidates_generated,
                "exploration_performed": result.exploration_performed,
                "processing_time": result.processing_time
            },
            "metadata": result.metadata
        }
    
    def _display_text_output(self, result: ReFoRCEResult):
        """Display result in text format"""
        print("\n" + "=" * 60)
        print("🎯 ReFoRCE Text-to-SQL Result")
        print("=" * 60)
        
        print(f"\n📝 Generated SQL ({result.pipeline_stage.upper()} stage):")
        print("```sql")
        print(result.final_sql)
        print("```")
        
        print(f"\n📊 Statistics:")
        print(f"• Confidence: {result.confidence:.1%}")
        print(f"• Processing Time: {result.processing_time:.2f}s")
        print(f"• Compression Ratio: {result.compression_ratio:.1%}")
        print(f"• Candidates Generated: {result.candidates_generated}")
        print(f"• Column Exploration: {'Yes' if result.exploration_performed else 'No'}")
        print(f"• Execution Ready: {'✓' if result.execution_successful else '✗'}")
        
        # Confidence indicator
        if result.confidence >= 0.8:
            print(f"\n✅ High confidence result")
        elif result.confidence >= 0.6:
            print(f"\n🔶 Medium confidence result")
        else:
            print(f"\n⚠️  Low confidence result - manual review recommended")
        
        print("=" * 60)
    
    def _result_to_dict(self, result: ReFoRCEResult) -> Dict[str, Any]:
        """Convert ReFoRCEResult to dictionary"""
        return {
            "sql": result.final_sql,
            "confidence": result.confidence,
            "pipeline_stage": result.pipeline_stage,
            "execution_successful": result.execution_successful,
            "compression_ratio": result.compression_ratio,
            "candidates_generated": result.candidates_generated,
            "exploration_performed": result.exploration_performed,
            "processing_time": result.processing_time
        }
    
    async def run_diagnostics(self):
        """Run system diagnostics"""
        print("🔧 ReFoRCE System Diagnostics")
        print("=" * 40)
        
        # Health check
        health_status = await self.workflow.health_check()
        print("\n📋 Component Health:")
        for component, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        # Configuration check
        print("\n⚙️  Configuration:")
        config_dict = settings.to_dict()
        for section, values in config_dict.items():
            print(f"  {section}:")
            for key, value in values.items():
                if 'password' in key.lower() or 'key' in key.lower():
                    value = "***" if value else "Not set"
                print(f"    {key}: {value}")
        
        # Database statistics
        try:
            db_manager = self.workflow.db_manager
            tables = db_manager.get_all_tables()
            print(f"\n📊 Database Statistics:")
            print(f"  Tables: {len(tables)}")
            
            if tables:
                table_sizes = db_manager.get_table_sizes()
                largest_table = max(table_sizes.items(), key=lambda x: x[1])
                print(f"  Largest table: {largest_table[0]} ({largest_table[1]:,} bytes)")
        except Exception as e:
            print(f"\n❌ Database connection failed: {e}")
        
        print("=" * 40)

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="ReFoRCE Text-to-SQL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                    # Interactive mode
  %(prog)s --request "Show all users"       # Single request
  %(prog)s --batch queries.txt              # Batch processing
  %(prog)s --diagnostics                    # System diagnostics
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    mode_group.add_argument(
        '--request', '-r',
        type=str,
        help='Process a single text-to-SQL request'
    )
    mode_group.add_argument(
        '--batch', '-b',
        type=str,
        help='Process requests from a file (one per line)'
    )
    mode_group.add_argument(
        '--diagnostics', '-d',
        action='store_true',
        help='Run system diagnostics'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    cli = ReFoRCECLI()
    
    try:
        # Initialize system
        await cli.initialize()
        
        # Execute based on mode
        if args.interactive:
            await cli.interactive_mode()
            
        elif args.request:
            result = await cli.process_single_request(args.request, args.format)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
                
        elif args.batch:
            result = await cli.process_batch_file(args.batch, args.output)
            if not args.output:
                print(f"Processed {result['total_requests']} requests")
                print(f"Successful: {result['successful']}")
                
        elif args.diagnostics:
            await cli.run_diagnostics()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())