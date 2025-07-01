#!/usr/bin/env python3
"""
Minimal MCP test - just test if the basic search works
"""

import requests
import json

# Direct test of MCP server
url = "http://localhost:8000/mcp/"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}

# Simple search test
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "intelligent_search",
        "arguments": {
            "search_request": {
                "query": "machine learning",
                "max_results": 1
            }
        }
    }
}

print("🔍 Testing MCP server...")
print(f"URL: {url}")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! MCP server is working!")
        search_result = result.get('result', {})
        print(f"Algorithm: {search_result.get('selected_algorithm')}")
        print(f"Results: {len(search_result.get('results', []))}")
    else:
        print(f"❌ Error {response.status_code}")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")