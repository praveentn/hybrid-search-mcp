# MCP Server Test Script
# Test the Hybrid RAG Search MCP Server deployed on Render

param(
    [string]$ServerUrl = "https://hybrid-search-mcp.onrender.com/mcp/"
)

# Set up headers for MCP protocol
$headers = @{
    'Content-Type' = 'application/json'
    'Accept' = 'application/json'
}

# Function to make MCP requests with proper error handling
function Invoke-MCPRequest {
    param(
        [string]$Method,
        [hashtable]$Params = @{},
        [string]$Id = "test-request"
    )
    
    $body = @{
        jsonrpc = "2.0"
        method = $Method
        id = $Id
    }
    
    if ($Params.Count -gt 0) {
        $body.params = $Params
    }
    
    $jsonBody = $body | ConvertTo-Json -Depth 10 -Compress
    
    try {
        Write-Host "📤 Request: $Method" -ForegroundColor Cyan
        Write-Host "   Body: $jsonBody" -ForegroundColor Gray
        
        $response = Invoke-WebRequest -Uri $ServerUrl -Method POST -Headers $headers -Body $jsonBody -ErrorAction Stop
        
        Write-Host "✅ Status: $($response.StatusCode)" -ForegroundColor Green
        
        $responseObj = $response.Content | ConvertFrom-Json
        return $responseObj
        
    } catch {
        Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode
            Write-Host "   Status Code: $statusCode" -ForegroundColor Red
            
            try {
                $errorBody = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($errorBody)
                $errorContent = $reader.ReadToEnd()
                Write-Host "   Response: $errorContent" -ForegroundColor Red
            } catch {
                Write-Host "   Could not read error response" -ForegroundColor Red
            }
        }
        return $null
    }
}

# Function to display results nicely
function Show-Result {
    param(
        [string]$TestName,
        [object]$Result
    )
    
    Write-Host "`n🔍 $TestName" -ForegroundColor Yellow
    Write-Host "=" * 50 -ForegroundColor Yellow
    
    if ($Result) {
        if ($Result.error) {
            Write-Host "❌ Error: $($Result.error.message)" -ForegroundColor Red
            Write-Host "   Code: $($Result.error.code)" -ForegroundColor Red
        } else {
            $Result | ConvertTo-Json -Depth 5 | Write-Host
        }
    } else {
        Write-Host "❌ No response received" -ForegroundColor Red
    }
}

# Start testing
Write-Host "🚀 Testing MCP Server: $ServerUrl" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor Magenta

# Test 1: Basic connectivity with initialize
Write-Host "`n1️⃣ Testing Initialize..." -ForegroundColor Blue
$initResult = Invoke-MCPRequest -Method "initialize" -Params @{
    protocolVersion = "2024-11-05"
    capabilities = @{
        roots = @{
            listChanged = $false
        }
        sampling = @{}
    }
    clientInfo = @{
        name = "test-client"
        version = "1.0.0"
    }
} -Id "init-1"

Show-Result "Initialize" $initResult

# Test 2: List tools
Write-Host "`n2️⃣ Testing Tools List..." -ForegroundColor Blue
$toolsResult = Invoke-MCPRequest -Method "tools/list" -Id "tools-1"
Show-Result "Tools List" $toolsResult

# Test 3: List resources  
Write-Host "`n3️⃣ Testing Resources List..." -ForegroundColor Blue
$resourcesResult = Invoke-MCPRequest -Method "resources/list" -Id "resources-1"
Show-Result "Resources List" $resourcesResult

# Test 4: Health check tool
Write-Host "`n4️⃣ Testing Health Check Tool..." -ForegroundColor Blue
$healthResult = Invoke-MCPRequest -Method "tools/call" -Params @{
    name = "get_server_health"
    arguments = @{}
} -Id "health-1"

Show-Result "Health Check" $healthResult

# Test 5: Server info tool
Write-Host "`n5️⃣ Testing Server Info Tool..." -ForegroundColor Blue
$infoResult = Invoke-MCPRequest -Method "tools/call" -Params @{
    name = "get_server_info"
    arguments = @{}
} -Id "info-1"

Show-Result "Server Info" $infoResult

# Test 6: Sample search test
Write-Host "`n6️⃣ Testing Sample Search..." -ForegroundColor Blue
$sampleResult = Invoke-MCPRequest -Method "tools/call" -Params @{
    name = "sample_search_test"
    arguments = @{}
} -Id "sample-1"

Show-Result "Sample Search Test" $sampleResult

# Test 7: Intelligent search
Write-Host "`n7️⃣ Testing Intelligent Search..." -ForegroundColor Blue
$searchResult = Invoke-MCPRequest -Method "tools/call" -Params @{
    name = "intelligent_search"
    arguments = @{
        search_request = @{
            query = "machine learning transformers"
            algorithm = "hybrid"
            max_results = 3
            explain = $true
        }
    }
} -Id "search-1"

Show-Result "Intelligent Search" $searchResult

# Test 8: Query intent analysis
Write-Host "`n8️⃣ Testing Query Intent Analysis..." -ForegroundColor Blue
$intentResult = Invoke-MCPRequest -Method "tools/call" -Params @{
    name = "analyze_query_intent" 
    arguments = @{
        query = "what is semantic search"
    }
} -Id "intent-1"

Show-Result "Query Intent Analysis" $intentResult

# Summary
Write-Host "`n📊 Test Summary" -ForegroundColor Magenta
Write-Host "=" * 30 -ForegroundColor Magenta

$tests = @("Initialize", "Tools List", "Resources List", "Health Check", "Server Info", "Sample Search", "Intelligent Search", "Intent Analysis")
$results = @($initResult, $toolsResult, $resourcesResult, $healthResult, $infoResult, $sampleResult, $searchResult, $intentResult)

for ($i = 0; $i -lt $tests.Count; $i++) {
    $status = if ($results[$i] -and -not $results[$i].error) { "✅ PASS" } else { "❌ FAIL" }
    Write-Host "$($i+1). $($tests[$i]): $status"
}

Write-Host "`n🎯 If tools/list works, your MCP server is functioning correctly!" -ForegroundColor Green
Write-Host "🔗 Server URL: $ServerUrl" -ForegroundColor Gray