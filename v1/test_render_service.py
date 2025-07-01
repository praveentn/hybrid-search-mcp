#!/usr/bin/env python3
"""
MCP Server Test Script (Python)
Comprehensive testing for the Hybrid RAG Search MCP Server deployed on Render
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys
import warnings

# Suppress SSL warnings when verification is disabled
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

@dataclass
class TestResult:
    name: str
    success: bool
    response: Optional[Dict[Any, Any]]
    error_message: Optional[str]
    duration_ms: float

class MCPTester:
    def __init__(self, server_url: str = "https://hybrid-search-mcp.onrender.com/mcp/", verify_ssl: bool = False):
        self.server_url = server_url
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'  # MCP requires both!
        })
        # Disable SSL verification for testing (can be enabled with verify_ssl=True)
        self.session.verify = verify_ssl
        
        self.test_results: List[TestResult] = []
        
    def make_mcp_request(self, method: str, params: Optional[Dict] = None, request_id: str = "test-request") -> TestResult:
        """Make an MCP JSON-RPC 2.0 request with proper error handling"""
        
        body = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        
        if params:
            body["params"] = params
            
        start_time = time.time()
        
        try:
            print(f"📤 Request: {method}")
            print(f"   Body: {json.dumps(body, indent=2)}")
            
            response = self.session.post(
                self.server_url,
                json=body,
                timeout=30
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            print(f"✅ Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                
                if "error" in response_data:
                    return TestResult(
                        name=method,
                        success=False,
                        response=response_data,
                        error_message=f"MCP Error: {response_data['error'].get('message', 'Unknown error')}",
                        duration_ms=duration_ms
                    )
                else:
                    return TestResult(
                        name=method,
                        success=True,
                        response=response_data,
                        error_message=None,
                        duration_ms=duration_ms
                    )
            else:
                error_text = response.text if response.text else "No response body"
                return TestResult(
                    name=method,
                    success=False,
                    response=None,
                    error_message=f"HTTP {response.status_code}: {error_text}",
                    duration_ms=duration_ms
                )
                
        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=method,
                success=False,
                response=None,
                error_message=f"Request failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    def show_result(self, result: TestResult):
        """Display test result in a formatted way"""
        print(f"\n🔍 {result.name}")
        print("=" * 50)
        print(f"⏱️  Duration: {result.duration_ms:.1f}ms")
        
        if result.success:
            print("✅ SUCCESS")
            if result.response:
                # Pretty print the response, but truncate if too long
                response_str = json.dumps(result.response, indent=2)
                if len(response_str) > 1000:
                    print(f"📊 Response (truncated):\n{response_str[:1000]}...")
                else:
                    print(f"📊 Response:\n{response_str}")
        else:
            print("❌ FAILED")
            print(f"🚫 Error: {result.error_message}")
            if result.response:
                print(f"📊 Response: {json.dumps(result.response, indent=2)}")
    
    def test_initialize(self) -> TestResult:
        """Test MCP initialize handshake"""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": False
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "python-test-client",
                "version": "1.0.0"
            }
        }
        
        result = self.make_mcp_request("initialize", params, "init-1")
        self.test_results.append(result)
        return result
    
    def test_tools_list(self) -> TestResult:
        """Test listing available tools"""
        result = self.make_mcp_request("tools/list", None, "tools-1")
        self.test_results.append(result)
        return result
    
    def test_resources_list(self) -> TestResult:
        """Test listing available resources"""
        result = self.make_mcp_request("resources/list", None, "resources-1")
        self.test_results.append(result)
        return result
    
    def test_health_check(self) -> TestResult:
        """Test server health check tool"""
        params = {
            "name": "get_server_health",
            "arguments": {}
        }
        
        result = self.make_mcp_request("tools/call", params, "health-1")
        self.test_results.append(result)
        return result
    
    def test_server_info(self) -> TestResult:
        """Test server info tool"""
        params = {
            "name": "get_server_info",
            "arguments": {}
        }
        
        result = self.make_mcp_request("tools/call", params, "info-1")
        self.test_results.append(result)
        return result
    
    def test_sample_search(self) -> TestResult:
        """Test sample search functionality"""
        params = {
            "name": "sample_search_test",
            "arguments": {}
        }
        
        result = self.make_mcp_request("tools/call", params, "sample-1")
        self.test_results.append(result)
        return result
    
    def test_intelligent_search(self) -> TestResult:
        """Test intelligent search with hybrid algorithm"""
        params = {
            "name": "intelligent_search",
            "arguments": {
                "search_request": {
                    "query": "machine learning transformers",
                    "algorithm": "hybrid",
                    "max_results": 3,
                    "explain": True
                }
            }
        }
        
        result = self.make_mcp_request("tools/call", params, "search-1")
        self.test_results.append(result)
        return result
    
    def test_query_intent_analysis(self) -> TestResult:
        """Test query intent analysis"""
        params = {
            "name": "analyze_query_intent",
            "arguments": {
                "query": "what is semantic search"
            }
        }
        
        result = self.make_mcp_request("tools/call", params, "intent-1")
        self.test_results.append(result)
        return result
    
    def test_algorithm_comparison(self) -> TestResult:
        """Test algorithm comparison functionality"""
        params = {
            "name": "compare_algorithms",
            "arguments": {
                "query": "vector database optimization",
                "max_results": 3
            }
        }
        
        result = self.make_mcp_request("tools/call", params, "compare-1")
        self.test_results.append(result)
        return result
    
    def test_add_document(self) -> TestResult:
        """Test adding a new document"""
        params = {
            "name": "add_document",
            "arguments": {
                "document": {
                    "id": "test_doc_python",
                    "title": "Python Test Document",
                    "content": "This is a test document added via Python script to verify the add_document functionality works correctly.",
                    "entities": ["python", "test", "document", "verification"],
                    "tags": ["testing", "python", "verification"]
                }
            }
        }
        
        result = self.make_mcp_request("tools/call", params, "add-doc-1")
        self.test_results.append(result)
        return result
    
    def test_search_analytics(self) -> TestResult:
        """Test search analytics functionality"""
        params = {
            "name": "get_search_analytics",
            "arguments": {}
        }
        
        result = self.make_mcp_request("tools/call", params, "analytics-1")
        self.test_results.append(result)
        return result
    
    def test_different_search_algorithms(self) -> List[TestResult]:
        """Test different search algorithms"""
        algorithms = ["keyword", "vector", "graph", "hybrid", "adaptive"]
        algorithm_results = []
        
        for algorithm in algorithms:
            params = {
                "name": "intelligent_search",
                "arguments": {
                    "search_request": {
                        "query": "neural networks",
                        "algorithm": algorithm,
                        "max_results": 2,
                        "explain": True
                    }
                }
            }
            
            result = self.make_mcp_request("tools/call", params, f"search-{algorithm}")
            result.name = f"Search ({algorithm})"
            algorithm_results.append(result)
            self.test_results.append(result)
        
        return algorithm_results
    
    def test_resource_access(self) -> List[TestResult]:
        """Test accessing different resources"""
        resources = [
            "search://documents",
            "search://algorithms", 
            "search://health"
        ]
        
        resource_results = []
        
        for resource in resources:
            params = {
                "uri": resource
            }
            
            result = self.make_mcp_request("resources/read", params, f"resource-{resource.split('/')[-1]}")
            result.name = f"Resource ({resource})"
            resource_results.append(result)
            self.test_results.append(result)
        
        return resource_results
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive MCP Server Testing")
        print(f"🔗 Server URL: {self.server_url}")
        print(f"🔐 SSL Verification: {'Enabled' if self.verify_ssl else 'Disabled'}")
        print("=" * 80)
        
        # CRITICAL: Initialize MCP session first
        print("\n🔧 MCP Protocol Initialization")
        print("-" * 40)
        
        init_result = self.test_initialize()
        self.show_result(init_result)
        
        if not init_result.success:
            print("\n🚨 CRITICAL: MCP initialization failed!")
            print("   Cannot proceed with other tests without proper initialization.")
            print("   Check server logs and MCP protocol compatibility.")
            self.show_summary()
            return
        
        print("\n✅ MCP Session Established - Proceeding with tests...")
        
        # Core MCP Protocol Tests
        print("\n🔧 Core MCP Protocol Tests")
        print("-" * 40)
        
        tools_result = self.test_tools_list()
        self.show_result(tools_result)
        
        resources_result = self.test_resources_list()
        self.show_result(resources_result)
        
        # Tool Functionality Tests
        print("\n🛠️  Tool Functionality Tests")
        print("-" * 40)
        
        health_result = self.test_health_check()
        self.show_result(health_result)
        
        info_result = self.test_server_info()
        self.show_result(info_result)
        
        sample_result = self.test_sample_search()
        self.show_result(sample_result)
        
        # Search Engine Tests
        print("\n🔍 Search Engine Tests")
        print("-" * 40)
        
        search_result = self.test_intelligent_search()
        self.show_result(search_result)
        
        intent_result = self.test_query_intent_analysis()
        self.show_result(intent_result)
        
        compare_result = self.test_algorithm_comparison()
        self.show_result(compare_result)
        
        # Document Management Tests
        print("\n📄 Document Management Tests")
        print("-" * 40)
        
        add_doc_result = self.test_add_document()
        self.show_result(add_doc_result)
        
        analytics_result = self.test_search_analytics()
        self.show_result(analytics_result)
        
        # Algorithm Tests
        print("\n🧠 Algorithm-Specific Tests")
        print("-" * 40)
        
        algorithm_results = self.test_different_search_algorithms()
        for result in algorithm_results:
            self.show_result(result)
        
        # Resource Tests
        print("\n📚 Resource Access Tests")
        print("-" * 40)
        
        resource_results = self.test_resource_access()
        for result in resource_results:
            self.show_result(result)
        
        # Summary
        self.show_summary()
    
    def show_summary(self):
        """Show comprehensive test summary"""
        print("\n📊 Test Summary")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"   • {result.name}: {result.error_message}")
        
        # Performance summary
        total_time = sum(r.duration_ms for r in self.test_results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        print(f"\n⏱️  Performance Summary:")
        print(f"   • Total Time: {total_time:.1f}ms")
        print(f"   • Average Response Time: {avg_time:.1f}ms")
        print(f"   • Fastest Response: {min(r.duration_ms for r in self.test_results):.1f}ms")
        print(f"   • Slowest Response: {max(r.duration_ms for r in self.test_results):.1f}ms")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        if passed_tests == total_tests:
            print("   🎉 All tests passed! Your MCP server is working perfectly.")
            print("   🚀 Ready for production use with AI clients.")
        elif passed_tests > total_tests * 0.8:
            print("   ⚠️  Most tests passed. Check failed tests for minor issues.")
            print("   🔧 Server is functional but may need some adjustments.")
        else:
            print("   🚨 Many tests failed. Server needs investigation.")
            print("   🔍 Check server logs and configuration.")
        
        print(f"\n🔗 Server URL: {self.server_url}")
        print("🔗 Documentation: https://your-server.onrender.com/docs")

def main():
    """Main test execution"""
    # Default server URL - can be overridden via command line
    server_url = "https://hybrid-search-mcp.onrender.com/mcp/"
    verify_ssl = False  # Default to False for testing convenience
    
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        if not server_url.endswith('/'):
            server_url += '/'
    
    # Check for SSL verification flag
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['true', 'ssl', 'verify']:
        verify_ssl = True
    
    print(f"🔐 SSL Verification: {'Enabled' if verify_ssl else 'Disabled (for testing)'}")
    if not verify_ssl:
        print("⚠️  Note: SSL verification disabled for testing. Enable with: python script.py <url> ssl")
    
    # Create tester and run comprehensive tests
    tester = MCPTester(server_url, verify_ssl)
    
    try:
        tester.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        tester.show_summary()
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check if server URL is correct")
        print("   • Verify server is running and accessible")
        print("   • Try with SSL verification disabled: python script.py <url> nossl")
        print("   • Test with curl first: curl -k https://your-server.com/mcp/")
        tester.show_summary()

if __name__ == "__main__":
    main()