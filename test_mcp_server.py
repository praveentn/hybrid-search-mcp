# File: test_mcp_server.py
"""
Test FastMCP client with SSL fixes for your MCP server
"""

import asyncio
import os
import ssl
import warnings
from fastmcp import Client

# Disable SSL warnings and verification for testing
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Try to disable SSL verification globally
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("✅ SSL verification disabled globally")
except Exception as e:
    print(f"⚠️  Could not disable SSL globally: {e}")

# Your server URL
url = "https://hybrid-search-mcp.onrender.com/mcp/"

async def test_mcp_server():
    """Test the MCP server with FastMCP client"""
    
    print(f"🔗 Connecting to: {url}")
    print("=" * 50)
    
    try:
        async with Client(url) as client:
            print("✅ Connected to MCP server!")
            
            # List available tools
            print("\n📋 Listing available tools...")
            tools = await client.list_tools()
            
            if tools:
                print(f"✅ Found {len(tools)} tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"   {i}. {tool.name}: {tool.description}")
            else:
                print("❌ No tools found")
                return
            
            # Test get_funny_oneliner tool
            print("\n🎭 Testing get_funny_oneliner tool...")
            test_names = ["Raj", "Priya", "Unni", "Maya"]
            
            for name in test_names:
                try:
                    result = await client.call_tool("get_funny_oneliner", {"name": name})
                    if result and len(result) > 0:
                        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                        print(f"   • {name}: {content}")
                    else:
                        print(f"   • {name}: No result returned")
                except Exception as e:
                    print(f"   • {name}: Error - {e}")
            
            # Test list_available_names tool
            print("\n📝 Testing list_available_names tool...")
            try:
                result = await client.call_tool("list_available_names", {})
                if result and len(result) > 0:
                    content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                    print(f"   {content}")
                else:
                    print("   No names list returned")
            except Exception as e:
                print(f"   Error listing names: {e}")
            
            print("\n🎉 All tests completed successfully!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure your server is running and accessible")
        print("2. Check if the URL is correct")
        print("3. Verify SSL certificates are working")
        print("4. Try running with: PYTHONHTTPSVERIFY=0 python test_fastmcp_fixed.py")

async def test_connection_only():
    """Just test if we can connect"""
    print("🔍 Testing basic connection...")
    
    try:
        async with Client(url) as client:
            print("✅ Basic connection successful!")
            return True
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False

async def main():
    print("🧪 FastMCP Client Test (SSL Fixed)")
    print("=" * 50)
    
    # First test basic connection
    if await test_connection_only():
        # If basic connection works, run full tests
        await test_mcp_server()
    else:
        print("\n💡 Connection failed. Your server might need the updated app.py")
        print("   Make sure you've deployed the updated version with MCP initialize support.")

if __name__ == "__main__":
    asyncio.run(main())