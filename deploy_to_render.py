#!/usr/bin/env python3
"""
Quick deployment helper for Render.com
"""

import os
import subprocess
import sys

def create_deployment_files():
    """Create all necessary files for Render deployment"""
    
    # 1. Update requirements.txt
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write("""fastmcp>=0.1.0
pydantic>=2.0.0
numpy>=1.24.0
uvicorn>=0.23.0
fastapi>=0.100.0
requests>=2.31.0
""")
    
    # 2. Create render.yaml for easy deployment
    with open("render.yaml", "w", encoding='utf-8') as f:
        f.write("""services:
  - type: web
    name: hybrid-search-mcp
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python hybrid_rag_search_mcp.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 8000
    healthCheckPath: /
""")
    
    # 3. Create .gitignore
    with open(".gitignore", "w", encoding='utf-8') as f:
        f.write("""__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
venv/
.DS_Store
*.log
""")
    
    # 4. Create startup script for cloud
    with open("start.py", "w", encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the MCP server
from hybrid_rag_search_mcp import mcp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}")
    mcp.run(transport="http", host="0.0.0.0", port=port)
""")
    
    print("Created deployment files:")
    print("   - requirements.txt")
    print("   - render.yaml") 
    print("   - .gitignore")
    print("   - start.py")

def create_git_repo():
    """Initialize git repo and commit files"""
    try:
        # Initialize git repo
        subprocess.run(["git", "init"], check=True)
        print("Git repository initialized")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("Files added to git")
        
        # Commit
        subprocess.run(["git", "commit", "-m", "Initial commit: Hybrid RAG Search MCP Server"], check=True)
        print("Initial commit created")
        
        print("\nNext steps:")
        print("1. Create a GitHub repository")
        print("2. Add remote: git remote add origin https://github.com/yourusername/hybrid-search-mcp.git")
        print("3. Push: git push -u origin main")
        print("4. Deploy on Render.com by connecting your GitHub repo")
        
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print("You can manually create the git repo")

def main():
    print("Hybrid RAG Search MCP - Deployment Helper")
    print("=" * 50)
    
    print("\nCreating deployment files...")
    create_deployment_files()
    
    print("\nSetting up git repository...")
    if input("Initialize git repo? (y/n): ").lower() == 'y':
        create_git_repo()
    
    print(f"\nReady for deployment!")
    print(f"Current directory: {os.getcwd()}")
    print("\nDeployment options:")
    print("1. Render.com - Upload files or connect GitHub")
    print("2. Railway - Use web interface")
    print("3. Vercel - Deploy with web interface")
    print("4. Replit - Import from GitHub")

if __name__ == "__main__":
    main()