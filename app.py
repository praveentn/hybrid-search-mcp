# app.py
import json
import logging
import os
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
from names_data import get_oneliner, get_all_names

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MCP Server metadata
MCP_SERVER_INFO = {
    "name": "funny-oneliner-server",
    "version": "1.0.0",
    "description": "An MCP server that provides funny one-liners for Indian and Kerala names",
    "author": "FastMCP Developer",
    "license": "MIT"
}

# Available tools
TOOLS = [
    {
        "name": "get_funny_oneliner",
        "description": "Get a funny one-liner for a given name",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name to get a funny one-liner for"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_available_names",
        "description": "Get a list of all available names in the database",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

def create_mcp_response(tools=None, resources=None, result=None, error=None):
    """Create a standard MCP response"""
    response = {
        "jsonrpc": "2.0",
        "id": 1
    }
    
    if tools is not None:
        response["result"] = {
            "capabilities": {
                "tools": {}
            },
            "serverInfo": MCP_SERVER_INFO,
            "tools": tools
        }
    elif result is not None:
        response["result"] = result
    elif error is not None:
        response["error"] = error
    
    return response

def stream_json_response(data):
    """Stream JSON response as required by MCP protocol"""
    def generate():
        yield json.dumps(data) + '\n'
    
    return Response(generate(), mimetype='application/json')

@app.route('/mcp/', methods=['POST'])
def mcp_endpoint():
    """Main MCP endpoint for HTTP transport"""
    try:
        request_data = request.get_json()
        logger.info(f"Received MCP request: {request_data}")
        
        if not request_data:
            # Initial connection - return server capabilities and tools
            response = create_mcp_response(tools=TOOLS)
            return stream_json_response(response)
        
        # Handle tool invocations
        method = request_data.get('method')
        params = request_data.get('params', {})
        
        if method == 'tools/call':
            tool_name = params.get('name')
            tool_arguments = params.get('arguments', {})
            
            if tool_name == 'get_funny_oneliner':
                name = tool_arguments.get('name', '').strip()
                if not name:
                    error_response = create_mcp_response(error={
                        "code": -1,
                        "message": "Name parameter is required"
                    })
                    return stream_json_response(error_response)
                
                oneliner = get_oneliner(name)
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": f"🎭 One-liner for {name.title()}: {oneliner}"
                        }
                    ]
                }
                response = create_mcp_response(result=result)
                return stream_json_response(response)
            
            elif tool_name == 'list_available_names':
                names = get_all_names()
                names_text = "📝 Available names: " + ", ".join([name.title() for name in sorted(names)])
                result = {
                    "content": [
                        {
                            "type": "text", 
                            "text": names_text
                        }
                    ]
                }
                response = create_mcp_response(result=result)
                return stream_json_response(response)
            
            else:
                error_response = create_mcp_response(error={
                    "code": -2,
                    "message": f"Unknown tool: {tool_name}"
                })
                return stream_json_response(error_response)
        
        elif method == 'tools/list':
            response = create_mcp_response(tools=TOOLS)
            return stream_json_response(response)
        
        else:
            error_response = create_mcp_response(error={
                "code": -3,
                "message": f"Unknown method: {method}"
            })
            return stream_json_response(error_response)
            
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {str(e)}")
        error_response = create_mcp_response(error={
            "code": -4,
            "message": f"Internal server error: {str(e)}"
        })
        return stream_json_response(error_response)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "server": MCP_SERVER_INFO})

# Web Interface
WEB_INTERFACE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Funny One-Liner MCP Server</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #5a67d8;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            background: #f8fafc;
        }
        .section h2 {
            color: #4a5568;
            margin-top: 0;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #cbd5e0;
            border-radius: 6px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background: #5a67d8;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: background-color 0.3s;
        }
        button:hover {
            background: #4c51bf;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #e6fffa;
            border: 2px solid #38b2ac;
            border-radius: 6px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .error {
            background: #fed7d7;
            border-color: #e53e3e;
            color: #c53030;
        }
        .endpoint-info {
            background: #ebf8ff;
            border: 2px solid #3182ce;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .endpoint-info code {
            background: #2d3748;
            color: #e2e8f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
        .names-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .name-tag {
            background: #bee3f8;
            padding: 8px 12px;
            border-radius: 20px;
            text-align: center;
            font-size: 14px;
            border: 1px solid #90cdf4;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Funny One-Liner MCP Server</h1>
        <p class="subtitle">Get hilarious one-liners for Indian and Kerala names!</p>
        
        <div class="endpoint-info">
            <strong>🔗 MCP Endpoint:</strong> <code>POST {{ base_url }}/mcp/</code><br>
            <strong>📊 Health Check:</strong> <code>GET {{ base_url }}/health</code>
        </div>

        <div class="section">
            <h2>🎪 Try the One-Liner Tool</h2>
            <input type="text" id="nameInput" placeholder="Enter a name (e.g., Raj, Priya, Unni)" />
            <button onclick="getOneLiner()">Get Funny One-Liner</button>
            <div id="result"></div>
        </div>

        <div class="section">
            <h2>📋 Available Names</h2>
            <button onclick="listNames()">Show All Available Names</button>
            <div id="namesList"></div>
        </div>

        <div class="section">
            <h2>🛠️ Test MCP Protocol</h2>
            <p>Use these JSON payloads to test the MCP endpoints:</p>
            <h3>Get Server Info & Tools:</h3>
            <pre style="background: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto;">POST {{ base_url }}/mcp/
Content-Type: application/json

{}</pre>
            
            <h3>Call Get One-Liner Tool:</h3>
            <pre style="background: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto;">POST {{ base_url }}/mcp/
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_funny_oneliner",
    "arguments": {
      "name": "Raj"
    }
  }
}</pre>
        </div>
    </div>

    <script>
        const baseUrl = window.location.origin;

        async function getOneLiner() {
            const name = document.getElementById('nameInput').value.trim();
            const resultDiv = document.getElementById('result');
            
            if (!name) {
                resultDiv.innerHTML = '<div class="result error">Please enter a name!</div>';
                return;
            }

            try {
                const response = await fetch('/mcp/', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "get_funny_oneliner",
                            "arguments": {"name": name}
                        }
                    })
                });

                const data = await response.json();
                if (data.result && data.result.content) {
                    resultDiv.innerHTML = '<div class="result">' + data.result.content[0].text + '</div>';
                } else if (data.error) {
                    resultDiv.innerHTML = '<div class="result error">Error: ' + data.error.message + '</div>';
                } else {
                    resultDiv.innerHTML = '<div class="result error">Unexpected response format</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="result error">Network error: ' + error.message + '</div>';
            }
        }

        async function listNames() {
            const namesDiv = document.getElementById('namesList');
            
            try {
                const response = await fetch('/mcp/', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "list_available_names",
                            "arguments": {}
                        }
                    })
                });

                const data = await response.json();
                if (data.result && data.result.content) {
                    const namesText = data.result.content[0].text;
                    const names = namesText.replace('📝 Available names: ', '').split(', ');
                    
                    let nameGrid = '<div class="names-grid">';
                    names.forEach(name => {
                        nameGrid += `<div class="name-tag">${name}</div>`;
                    });
                    nameGrid += '</div>';
                    
                    namesDiv.innerHTML = nameGrid;
                } else if (data.error) {
                    namesDiv.innerHTML = '<div class="result error">Error: ' + data.error.message + '</div>';
                }
            } catch (error) {
                namesDiv.innerHTML = '<div class="result error">Network error: ' + error.message + '</div>';
            }
        }

        // Allow Enter key to trigger search
        document.getElementById('nameInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                getOneLiner();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def web_interface():
    """Serve the web interface"""
    base_url = request.url_root.rstrip('/')
    return render_template_string(WEB_INTERFACE_HTML, base_url=base_url)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
