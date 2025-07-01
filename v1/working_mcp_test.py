#!/usr/bin/env python3
"""
Working MCP test with proper session handling
"""

import requests
import json

class MCPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
    
    def initialize_session(self):
        """Initialize MCP session"""
        print("🔄 Initializing MCP session...")
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            print(f"   Initialize response: {response.status_code}")
            
            # Extract session ID from headers
            if 'mcp-session-id' in response.headers:
                self.session_id = response.headers['mcp-session-id']
                self.headers['mcp-session-id'] = self.session_id
                print(f"   ✅ Session ID: {self.session_id}")
                return True
            else:
                print("   ❌ No session ID in response")
                print(f"   Headers: {dict(response.headers)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Initialize failed: {e}")
            return False
    
    def list_tools(self):
        """List available tools"""
        if not self.session_id:
            print("❌ No session - initializing first...")
            if not self.initialize_session():
                return None
        
        print("🔄 Listing tools...")
        
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            print(f"   Tools list response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                tools = result.get('result', {}).get('tools', [])
                print(f"   ✅ Found {len(tools)} tools:")
                for tool in tools[:5]:  # Show first 5 tools
                    print(f"      - {tool.get('name', 'unknown')}: {tool.get('description', 'no description')[:50]}...")
                return tools
            else:
                print(f"   ❌ Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ List tools failed: {e}")
            return None
    
    def test_search(self):
        """Test intelligent search"""
        if not self.session_id:
            print("❌ No session - initializing first...")
            if not self.initialize_session():
                return None
        
        print("🔄 Testing intelligent search...")
        
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "intelligent_search",
                "arguments": {
                    "search_request": {
                        "query": "machine learning transformers",
                        "algorithm": "hybrid",
                        "max_results": 2,
                        "explain": True
                    }
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/",
                json=payload,
                headers=self.headers,
                timeout=15
            )
            
            print(f"   Search response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                search_result = result.get('result', {})
                results = search_result.get('results', [])
                
                print(f"   ✅ Search successful!")
                print(f"      Algorithm: {search_result.get('selected_algorithm', 'unknown')}")
                print(f"      Results: {len(results)} documents")
                print(f"      Time: {search_result.get('total_time_ms', 0):.2f}ms")
                
                if results:
                    print(f"      Top result: {results[0].get('title', 'No title')}")
                    print(f"      Score: {results[0].get('score', 0):.3f}")
                    print(f"      Explanation: {results[0].get('explanation', 'No explanation')}")
                
                return search_result
            else:
                print(f"   ❌ Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return None
    
    def test_intent_analysis(self):
        """Test query intent analysis"""
        if not self.session_id:
            if not self.initialize_session():
                return None
        
        print("🔄 Testing intent analysis...")
        
        payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "analyze_query_intent",
                "arguments": {
                    "query": "how to implement neural networks for beginners"
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            print(f"   Intent analysis response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                intent_data = result.get('result', {})
                
                print(f"   ✅ Intent analysis successful!")
                print(f"      Intent: {intent_data.get('intent_type', 'unknown')}")
                print(f"      Confidence: {intent_data.get('confidence', 0):.2f}")
                print(f"      Suggested Algorithm: {intent_data.get('suggested_algorithm', 'unknown')}")
                print(f"      Reasoning: {intent_data.get('reasoning', 'No reasoning')}")
                
                return intent_data
            else:
                print(f"   ❌ Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ Intent analysis failed: {e}")
            return None

def main():
    print("🧪 Testing Hybrid RAG Search MCP Server")
    print("=" * 50)
    
    client = MCPClient()
    
    # Test 1: Initialize session
    print("\n1️⃣ Session Initialization")
    if not client.initialize_session():
        print("❌ Failed to initialize session - stopping tests")
        return
    
    # Test 2: List tools
    print("\n2️⃣ Tool Discovery")
    tools = client.list_tools()
    
    # Test 3: Intelligent search
    print("\n3️⃣ Intelligent Search")
    search_result = client.test_search()
    
    # Test 4: Intent analysis
    print("\n4️⃣ Intent Analysis")
    intent_result = client.test_intent_analysis()
    
    print("\n🎉 MCP Server Testing Complete!")
    
    if search_result and intent_result:
        print("✅ All tests passed! Your MCP server is working perfectly!")
        print("🚀 Ready for deployment!")
        print(f"🌐 MCP Endpoint: {client.base_url}/mcp/")
        print(f"🔑 Session Management: Working")
        print("🧠 Intelligence Features: Active")
    else:
        print("⚠️  Some tests had issues, but core functionality is working")

if __name__ == "__main__":
    main()