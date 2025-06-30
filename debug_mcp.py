#!/usr/bin/env python3
"""
Simple debug script for MCP server
"""

import requests
import json

def debug_mcp_server():
    """Debug MCP server with detailed logging"""
    
    base_url = "http://localhost:8000"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    print("🔍 Debugging MCP Server...")
    print(f"📍 URL: {base_url}/mcp/")
    
    # Test 1: Basic connectivity
    print("\n1️⃣ Testing basic connectivity...")
    try:
        response = requests.get(f"{base_url}", timeout=5)
        print(f"   GET /: {response.status_code}")
        if response.status_code != 404:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ GET / failed: {e}")
    
    # Test 2: MCP endpoint basic post
    print("\n2️⃣ Testing MCP endpoint...")
    try:
        response = requests.post(f"{base_url}/mcp/", timeout=5)
        print(f"   POST /mcp/ (no data): {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ POST /mcp/ failed: {e}")
    
    # Test 3: JSON-RPC tools list
    print("\n3️⃣ Testing tools list...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post(
            f"{base_url}/mcp/",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        print(f"   tools/list: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        print(f"   Response: {response.text[:500]}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                tools = result.get('result', {}).get('tools', [])
                print(f"   ✅ Found {len(tools)} tools")
                for tool in tools[:3]:  # Show first 3 tools
                    print(f"      - {tool.get('name', 'unknown')}")
            except:
                print("   ❌ Could not parse JSON response")
        
    except Exception as e:
        print(f"   ❌ tools/list failed: {e}")
    
    # Test 4: Simple search
    print("\n4️⃣ Testing simple search...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "intelligent_search",
                "arguments": {
                    "search_request": {
                        "query": "test",
                        "max_results": 1,
                        "explain": False
                    }
                }
            }
        }
        
        response = requests.post(
            f"{base_url}/mcp/",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        print(f"   intelligent_search: {response.status_code}")
        print(f"   Response: {response.text[:300]}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                search_result = result.get('result', {})
                print(f"   ✅ Search completed: {search_result.get('selected_algorithm', 'unknown')} algorithm")
            except:
                print("   ❌ Could not parse search response")
        
    except Exception as e:
        print(f"   ❌ intelligent_search failed: {e}")
    
    print("\n🎯 Debug complete!")

if __name__ == "__main__":
    debug_mcp_server()