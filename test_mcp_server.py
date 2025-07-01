# File: test_mcp_server.py
#!/usr/bin/env python3
"""
Test script for the Funny One-Liner MCP Server
"""

import requests
import json
import sys
from typing import Dict, Any

class MCPServerTester:
    def __init__(self, base_url: str):
        """Initialize the tester with the server base URL"""
        self.base_url = base_url.rstrip('/')
        self.mcp_url = f"{self.base_url}/mcp/"
        self.health_url = f"{self.base_url}/health"
        
    def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        print("🔍 Testing health check endpoint...")
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_mcp_capabilities(self) -> bool:
        """Test MCP server capabilities discovery"""
        print("\n🔍 Testing MCP capabilities discovery...")
        try:
            # Send empty POST request to get server capabilities
            response = requests.post(
                self.mcp_url,
                json={},
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ MCP capabilities response received:")
                print(f"   Server: {data.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
                print(f"   Version: {data.get('result', {}).get('serverInfo', {}).get('version', 'Unknown')}")
                
                tools = data.get('result', {}).get('tools', [])
                print(f"   Available tools: {len(tools)}")
                for tool in tools:
                    print(f"     - {tool.get('name')}: {tool.get('description')}")
                return True
            else:
                print(f"❌ MCP capabilities test failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ MCP capabilities test error: {e}")
            return False
    
    def test_get_oneliner_tool(self, name: str = "Raj") -> bool:
        """Test the get_funny_oneliner tool"""
        print(f"\n🔍 Testing get_funny_oneliner tool with name '{name}'...")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_funny_oneliner",
                    "arguments": {
                        "name": name
                    }
                }
            }
            
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'content' in data['result']:
                    oneliner = data['result']['content'][0]['text']
                    print(f"✅ One-liner received: {oneliner}")
                    return True
                else:
                    print(f"❌ Unexpected response format: {data}")
                    return False
            else:
                print(f"❌ Get oneliner test failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Get oneliner test error: {e}")
            return False
    
    def test_list_names_tool(self) -> bool:
        """Test the list_available_names tool"""
        print("\n🔍 Testing list_available_names tool...")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "list_available_names",
                    "arguments": {}
                }
            }
            
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'content' in data['result']:
                    names_text = data['result']['content'][0]['text']
                    print(f"✅ Names list received: {names_text[:100]}...")
                    return True
                else:
                    print(f"❌ Unexpected response format: {data}")
                    return False
            else:
                print(f"❌ List names test failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ List names test error: {e}")
            return False
    
    def test_invalid_tool(self) -> bool:
        """Test calling an invalid tool (should return error)"""
        print("\n🔍 Testing invalid tool call (should fail gracefully)...")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "nonexistent_tool",
                    "arguments": {}
                }
            }
            
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"✅ Error handling works: {data['error']['message']}")
                    return True
                else:
                    print(f"❌ Expected error response, got: {data}")
                    return False
            else:
                print(f"❌ Invalid tool test failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Invalid tool test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print(f"🚀 Starting MCP Server Tests for: {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("MCP Capabilities", self.test_mcp_capabilities),
            ("Get One-Liner Tool", lambda: self.test_get_oneliner_tool("Priya")),
            ("List Names Tool", self.test_list_names_tool),
            ("Invalid Tool Error Handling", self.test_invalid_tool),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        
        print("\n" + "="*60)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your MCP server is working correctly.")
            return True
        else:
            print(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
            return False

def main():
    """Main function to run the tests"""
    if len(sys.argv) != 2:
        print("Usage: python test_mcp_server.py <server_url>")
        print("Example: python test_mcp_server.py https://your-app.onrender.com")
        print("Example: python test_mcp_server.py http://localhost:8000")
        sys.exit(1)
    
    server_url = sys.argv[1]
    tester = MCPServerTester(server_url)
    
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🎯 Quick Test Commands:")
        print(f"   Health Check: curl {server_url}/health")
        print(f"   Web Interface: {server_url}/")
        print(f"   MCP Endpoint: curl -X POST {server_url}/mcp/ -H 'Content-Type: application/json' -d '{{}}'")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
