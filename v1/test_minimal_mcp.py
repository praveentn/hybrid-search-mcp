#!/usr/bin/env python3
"""
Minimal MCP Test - Just the basics
Tests only initialization and tools list to verify MCP protocol is working
"""

import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def minimal_mcp_test():
    """Test minimal MCP protocol flow"""
    
    server_url = "https://hybrid-search-mcp.onrender.com/mcp/"
    
    # Create session
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'
    })
    
    print("🧪 Minimal MCP Protocol Test")
    print(f"🔗 {server_url}")
    print("=" * 40)
    
    # Step 1: Initialize
    print("\n1️⃣ Initialize MCP...")
    try:
        response = session.post(server_url, json={
            "jsonrpc": "2.0",
            "method": "initialize", 
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": "minimal-test", "version": "1.0"}
            },
            "id": "init"
        }, timeout=10)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            if 'result' in data:
                print("✅ Initialize SUCCESS")
            else:
                print("❌ Initialize FAILED")
                return
        else:
            print(f"❌ HTTP Error: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return
    
    # Step 2: Send initialized notification
    print("\n2️⃣ Send Initialized Notification...")
    try:
        response = session.post(server_url, json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }, timeout=10)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Initialized notification sent")
        else:
            print(f"⚠️  Notification status: {response.status_code} (may be normal)")
            
    except Exception as e:
        print(f"⚠️  Notification exception: {e} (may be normal)")
    
    # Step 3: List Tools
    print("\n3️⃣ List Tools...")
    try:
        response = session.post(server_url, json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": "tools"
        }, timeout=10)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            if 'result' in data and 'tools' in data['result']:
                tools = data['result']['tools']
                print(f"✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"   • {tool['name']}")
            else:
                print("❌ Tools list FAILED")
        else:
            print(f"❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n🎯 If both steps succeeded, MCP server is working!")

if __name__ == "__main__":
    minimal_mcp_test()