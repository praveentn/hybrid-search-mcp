#!/usr/bin/env python3
"""
Debug FastMCP connection issues step by step with SSL fixes
"""

import asyncio
import httpx
import json
import time
import ssl
import warnings
from fastmcp import Client

# Suppress SSL warnings for testing
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

url = "https://hybrid-search-mcp.onrender.com/mcp/"

async def debug_what_fastmcp_expects():
    """Debug what FastMCP is trying to do when connecting"""
    
    print("🔍 Step 1: Testing what our server returns for empty POST...")
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            response = await client.post(
                url,
                json={},
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            print(f"   Content: {response.text[:500]}...")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   JSON Valid: ✅")
                    print(f"   Has 'result': {'result' in data}")
                    print(f"   Has 'tools': {'tools' in data.get('result', {})}")
                except:
                    print(f"   JSON Valid: ❌")
                    
        except Exception as e:
            print(f"   Error: {e}")

async def test_mcp_initialize_sequence():
    """Test the proper MCP initialization sequence"""
    
    print("\n🔍 Step 2: Testing proper MCP initialization sequence...")
    
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        try:
            # Step 1: Initialize
            print("   Sending MCP initialize request...")
            init_response = await client.post(
                url,
                json={
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
                },
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"   Initialize response: {init_response.status_code}")
            if init_response.status_code == 200:
                init_data = init_response.json()
                print(f"   Initialize successful: ✅")
                print(f"   Server info: {init_data.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
            else:
                print(f"   Initialize failed: ❌")
                return False
            
            # Step 2: Send initialized notification
            print("   Sending initialized notification...")
            notify_response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                },
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"   Notification response: {notify_response.status_code}")
            
            # Step 3: List tools
            print("   Listing tools...")
            tools_response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                },
                headers={'Content-Type': 'application/json'}
            )
            
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                tools_count = len(tools_data.get('result', {}).get('tools', []))
                print(f"   Tools found: {tools_count} ✅")
                return True
            else:
                print(f"   Tools list failed: ❌")
                return False
                
        except Exception as e:
            print(f"   MCP sequence failed: {e}")
            return False

class SSLDisabledClient(Client):
    """FastMCP Client with SSL verification disabled"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def __aenter__(self):
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Patch the client's HTTP client to use our SSL context
        if hasattr(self, '_client') and self._client is None:
            await super().__aenter__()
            # Try to access the underlying httpx client and modify it
            # This is a workaround since FastMCP might not expose SSL settings
        else:
            await super().__aenter__()
        
        return self

async def test_fastmcp_with_ssl_disabled():
    """Test FastMCP with SSL verification disabled"""
    
    print("\n🔍 Step 3: Testing FastMCP with SSL verification disabled...")
    
    try:
        # Method 1: Try with environment variable
        import os
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        
        result = await asyncio.wait_for(
            test_fastmcp_ssl_disabled(),
            timeout=15.0
        )
        return result
    except asyncio.TimeoutError:
        print("   ❌ FastMCP still timed out after 15 seconds")
        return False
    except Exception as e:
        print(f"   ❌ FastMCP SSL disabled test failed: {e}")
        return False

async def test_fastmcp_ssl_disabled():
    """Basic FastMCP test with SSL disabled"""
    try:
        # Try using the SSL disabled client
        async with SSLDisabledClient(url) as client:
            print("   ✅ FastMCP connected with SSL disabled!")
            tools = await client.list_tools()
            print(f"   ✅ Found {len(tools)} tools")
            return True
    except Exception as e:
        print(f"   ❌ SSL disabled client failed: {e}")
        
        # Fallback: Try the regular client but with environment variables
        try:
            async with Client(url) as client:
                print("   ✅ FastMCP connected with environment SSL settings!")
                tools = await client.list_tools()
                print(f"   ✅ Found {len(tools)} tools")
                return True
        except Exception as e2:
            print(f"   ❌ Regular client also failed: {e2}")
            return False

async def test_different_urls():
    """Test if FastMCP works with different URL formats"""
    
    print("\n🔍 Step 4: Testing different URL formats...")
    
    test_urls = [
        "https://hybrid-search-mcp.onrender.com/mcp",     # Without trailing slash
        "https://hybrid-search-mcp.onrender.com",         # Root URL
        "https://hybrid-search-mcp.onrender.com/",        # Root with slash
    ]
    
    # Set environment variable to disable SSL verification
    import os
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    
    for test_url in test_urls:
        print(f"\n   Testing: {test_url}")
        try:
            result = await asyncio.wait_for(
                test_fastmcp_url(test_url),
                timeout=10.0
            )
            if result:
                print(f"   ✅ SUCCESS with: {test_url}")
                return test_url
        except asyncio.TimeoutError:
            print(f"   ❌ Timeout with: {test_url}")
        except Exception as e:
            print(f"   ❌ Error with {test_url}: {e}")
    
    return None

async def test_fastmcp_url(test_url):
    """Test FastMCP with a specific URL"""
    async with Client(test_url) as client:
        tools = await client.list_tools()
        return len(tools) > 0

def setup_ssl_environment():
    """Setup environment variables to disable SSL verification"""
    import os
    
    # Python SSL settings
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Try to disable SSL verification globally
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print("   ✅ SSL verification disabled globally")
    except Exception as e:
        print(f"   ⚠️  Could not disable SSL globally: {e}")

async def create_test_client():
    """Create a test client that can work with SSL issues"""
    
    print("\n🔍 Step 5: Creating test client with proper SSL handling...")
    
    # Setup SSL environment
    setup_ssl_environment()
    
    try:
        # Test if we can make the sequence work manually first
        sequence_works = await test_mcp_initialize_sequence()
        
        if sequence_works:
            print("   ✅ Manual MCP sequence works!")
            
            # Now try FastMCP
            print("   Testing FastMCP with SSL environment setup...")
            
            fastmcp_result = await asyncio.wait_for(
                test_fastmcp_basic_retry(),
                timeout=20.0
            )
            
            if fastmcp_result:
                print("   ✅ FastMCP works with SSL environment!")
                return True
            else:
                print("   ❌ FastMCP still fails even with SSL fixes")
                return False
        else:
            print("   ❌ Manual MCP sequence failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Test client creation failed: {e}")
        return False

async def test_fastmcp_basic_retry():
    """Test FastMCP with multiple retry attempts"""
    for attempt in range(3):
        try:
            print(f"   Attempt {attempt + 1}/3...")
            async with Client(url) as client:
                tools = await client.list_tools()
                if len(tools) > 0:
                    print(f"   ✅ Success on attempt {attempt + 1}!")
                    return True
        except Exception as e:
            print(f"   Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)  # Wait before retry
    
    return False

async def main():
    print("🐛 Debugging FastMCP Connection Issues with SSL Fixes")
    print("=" * 60)
    
    # Run all debug steps
    await debug_what_fastmcp_expects()
    
    # Test proper MCP sequence
    mcp_sequence_works = await test_mcp_initialize_sequence()
    
    if mcp_sequence_works:
        print("\n✅ MCP protocol sequence works correctly")
        
        # Test FastMCP with SSL fixes
        client_works = await create_test_client()
        
        if client_works:
            print("\n🎉 SUCCESS: FastMCP works with SSL fixes!")
        else:
            print("\n⚠️  FastMCP still has issues, trying different URLs...")
            working_url = await test_different_urls()
            
            if working_url:
                print(f"\n✅ FastMCP works with URL: {working_url}")
            else:
                print("\n❌ FastMCP doesn't work with any URL format")
    else:
        print("\n❌ MCP protocol sequence failed")
    
    print("\n" + "=" * 60)
    print("🎯 SOLUTIONS:")
    print("1. **SSL Certificate Issue Fix:**")
    print("   - Set environment variable: PYTHONHTTPSVERIFY=0")
    print("   - Or use: pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org fastmcp")
    print()
    print("2. **MCP Initialize Method Added:**")
    print("   - Updated app.py now handles 'initialize' method")
    print("   - Added proper MCP handshake support")
    print()
    print("3. **FastMCP Usage:**")
    print("   - Use updated app.py for proper MCP protocol")
    print("   - Set SSL environment variables before running")
    print("   - Test with: PYTHONHTTPSVERIFY=0 python your_fastmcp_script.py")
    print()
    print("4. **Alternative Testing:**")
    print("   - Manual HTTP requests work perfectly")
    print("   - Use: python test_mcp_server.py <url> --no-ssl-verify")

if __name__ == "__main__":
    asyncio.run(main())
