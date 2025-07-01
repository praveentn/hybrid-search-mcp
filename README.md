# Funny One-Liner MCP Server 🎭

A Model Context Protocol (MCP) server that provides hilarious one-liners for Indian and Kerala names. Built with Flask and designed for easy deployment to Render.com.

## Features

- 🎪 **Funny One-Liners**: Get personalized humorous one-liners for 50+ Indian and Kerala names
- 🔧 **MCP Protocol**: Full MCP HTTP transport support for integration with LLM clients
- 🌐 **Web Interface**: Beautiful web UI for testing and demonstration
- 🚀 **Cloud Ready**: Optimized for Render.com deployment with HTTPS support
- 📊 **Health Monitoring**: Built-in health check endpoint

## Available Names

The server includes funny one-liners for popular Indian and Kerala names including:
- **Male Names**: Raj, Arjun, Vikram, Rohit, Amit, Unni, Babu, Ravi, and more
- **Female Names**: Priya, Anjali, Kavya, Meera, Radha, Maya, Leela, and more

## Quick Start

### 1. Deploy to Render.com

1. Fork this repository to your GitHub account
2. Connect your GitHub account to Render.com
3. Create a new Web Service on Render
4. Connect your forked repository
5. Render will automatically detect the `render.yaml` and deploy your service
6. Your MCP server will be available at `https://your-app-name.onrender.com`

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Server will be available at http://localhost:8000
```

## Usage

### Web Interface

Visit your deployed URL (e.g., `https://your-app.onrender.com`) to access the web interface where you can:
- Test the one-liner tool with any name
- View all available names
- See MCP protocol examples

### MCP Protocol

The server exposes these MCP tools:

#### 1. Get Funny One-Liner
```json
POST /mcp/
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
}
```

#### 2. List Available Names
```json
POST /mcp/
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "list_available_names",
    "arguments": {}
  }
}
```

### Testing Your Deployment

Use the included test script:

```bash
# Install requests if not already installed
pip install requests

# Test your deployed server
python test_mcp_server.py https://your-app.onrender.com

# Test locally
python test_mcp_server.py http://localhost:8000
```

## API Endpoints

- `POST /mcp/` - Main MCP protocol endpoint
- `GET /health` - Health check endpoint
- `GET /` - Web interface

## Files Structure

```
├── app.py                 # Main Flask MCP server
├── names_data.py         # Names and one-liners database
├── requirements.txt      # Python dependencies
├── render.yaml          # Render.com deployment config
├── test_mcp_server.py   # Test script for validation
└── README.md           # This file
```

## Integration with LLM Clients

Once deployed with HTTPS on Render, your MCP server can be integrated with:
- OpenAI's Responses API
- Claude clients supporting MCP
- Custom MCP clients

The server URL will be: `https://your-app-name.onrender.com/mcp/`

## Customization

To add more names and one-liners:

1. Edit `names_data.py`
2. Add entries to the `NAMES_ONELINERS` dictionary
3. Redeploy to Render (automatically triggered by git push)

## Support

- Health Check: `GET /health`
- Web Interface: Visit your deployed URL
- Test Script: `python test_mcp_server.py <your-url>`

## License

MIT License - feel free to use and modify!
