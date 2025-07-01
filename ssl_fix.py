#!/usr/bin/env python3
"""
Quick MCP Server Test - SSL Fix Version
Simple test script that bypasses SSL verification issues
"""

import requests
import json
import warnings

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_mcp_server_quick():
    """Quick test of MCP server with proper MCP protocol flow"""
    
    server_url = "https://hybrid-search-mcp.onrender.com/mcp/"
    
    # Create session with SSL verification disabled
    session = requests.Session()
    session.verify = False  # Disable SSL verification
    session.headers.update({
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'  # MCP requires both!
    })
    
    print("🚀 Quick MCP Server Test")
    print(f"🔗 Server: {server_url}")
    print("🔐 SSL Verification: DISABLED (for testing)")
    print("=" * 50)
    
    # Step 0: Initialize MCP session (REQUIRED FIRST)
    print("\n0️⃣ Initializing MCP Session...")
    try:
        init_response = session.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": False},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "quick-test-client",
                        "version": "1.0.0"
                    }
                },
                "id": "init-1"
            },
            timeout=10
        )
        
        print(f"   Status: {init_response.status_code}")
        if init_response.status_code == 200:
            init_data = init_response.json()
            if 'result' in init_data:
                print("   ✅ MCP session initialized successfully")
                # Get session info if available
                capabilities = init_data['result'].get('capabilities', {})
                print(f"   Server capabilities: {list(capabilities.keys())}")
            else:
                print(f"   ❌ Init failed: {init_data}")
                return
        else:
            print(f"   ❌ Init HTTP Error: {init_response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Init Error: {e}")
        return
    
    # Test 1: Tools List
    print("\n1️⃣ Testing Tools List...")
    try:
        response = session.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": "test-1"
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and 'tools' in data['result']:
                tools = data['result']['tools']
                print(f"   ✅ Found {len(tools)} tools")
                for tool in tools[:3]:  # Show first 3 tools
                    print(f"      • {tool['name']}")
                if len(tools) > 3:
                    print(f"      • ... and {len(tools)-3} more")
            else:
                print(f"   ❌ Unexpected response: {data}")
        else:
            print(f"   ❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health Check
    print("\n2️⃣ Testing Health Check...")
    try:
        response = session.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_server_health",
                    "arguments": {}
                },
                "id": "test-2"
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                print("   ✅ Health check successful")
                result = data['result']
                if isinstance(result, dict) and 'content' in result:
                    content = result['content'][0]['text']
                    print(f"      Status: {content.get('status', 'unknown')}")
                    print(f"      Uptime: {content.get('server_info', {}).get('uptime_seconds', 'unknown')}s")
            else:
                print(f"   ❌ Unexpected response: {data}")
        else:
            print(f"   ❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Simple Search
    print("\n3️⃣ Testing Sample Search...")
    try:
        response = session.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "sample_search_test",
                    "arguments": {}
                },
                "id": "test-3"
            },
            timeout=15
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                print("   ✅ Sample search successful")
                result = data['result']
                if isinstance(result, dict) and 'content' in result:
                    content = result['content'][0]['text']
                    print(f"      Test Summary: {content.get('test_summary', 'unknown')}")
                    print(f"      System Status: {content.get('system_status', 'unknown')}")
            else:
                print(f"   ❌ Unexpected response: {data}")
        else:
            print(f"   ❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Intelligent Search
    print("\n4️⃣ Testing Intelligent Search...")
    try:
        response = session.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "intelligent_search",
                    "arguments": {
                        "search_request": {
                            "query": "machine learning",
                            "algorithm": "hybrid",
                            "max_results": 2,
                            "explain": True
                        }
                    }
                },
                "id": "test-4"
            },
            timeout=15
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                print("   ✅ Intelligent search successful")
                result = data['result']
                if isinstance(result, dict) and 'content' in result:
                    content = result['content'][0]['text']
                    results = content.get('results', [])
                    print(f"      Found {len(results)} results")
                    algorithm = content.get('selected_algorithm', 'unknown')
                    print(f"      Algorithm: {algorithm}")
                    if results:
                        print(f"      Top result: {results[0].get('title', 'unknown')}")
            else:
                print(f"   ❌ Unexpected response: {data}")
        else:
            print(f"   ❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎯 Quick Test Complete!")
    print("   ✅ If initialization and tools list passed, your MCP server is working!")
    print("   🚀 The server is ready for AI client integration.")
    print("   📝 Note: MCP protocol requires initialization before other calls.")

if __name__ == "__main__":
    test_mcp_server_quick()