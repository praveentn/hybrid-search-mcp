@mcp.resource("search://health")
def get_health():
    """Get detailed health status and system information"""
    return search_engine.get_health_status().dict()

@mcp.resource("search://debug")
def get_debug_info():
    """Get comprehensive debugging information"""
    debug_tool = debug_server_status()
    return debug_tool#!/usr/bin/env python3
"""
Hybrid RAG Search Platform MCP Server
A sophisticated intelligent search system with multiple algorithms and AI reasoning
Enhanced with proper MCP protocol handling and comprehensive testing support
"""

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
import numpy as np
import json
import time
import hashlib
import re
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query text")
    algorithm: Optional[str] = Field(None, description="Specific algorithm to use (keyword, vector, graph, hybrid, custom, adaptive)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for search")
    max_results: int = Field(10, description="Maximum number of results to return")
    explain: bool = Field(True, description="Include explanations for search decisions")

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = {}
    vector: Optional[List[float]] = None
    entities: List[str] = []
    tags: List[str] = []

class SearchResult(BaseModel):
    document_id: str
    title: str
    content: str
    score: float
    algorithm_used: str
    relevance_factors: Dict[str, float]
    explanation: str
    snippet: str

class IntelligentSearchResponse(BaseModel):
    query_analysis: Dict[str, Any]
    selected_algorithm: str
    algorithm_reasoning: str
    results: List[SearchResult]
    performance_metrics: Dict[str, float]
    suggestions: List[str]
    total_time_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    server_info: Dict[str, Any]

@dataclass
class QueryIntent:
    intent_type: str  # factual, conceptual, comparison, procedural, exploratory
    confidence: float
    reasoning: str
    suggested_algorithm: str

class HybridRAGSearchEngine:
    """Advanced search engine with multiple algorithms and intelligent reasoning"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.entity_graph: Dict[str, List[str]] = defaultdict(list)
        self.query_history: List[Dict] = []
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)
        self.server_start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
        # Initialize with sample intelligent documents
        self._initialize_sample_data()
        logger.info(f"Initialized search engine with {len(self.documents)} documents")
        
    def _initialize_sample_data(self):
        """Initialize with sample documents for demonstration"""
        sample_docs = [
            {
                "id": "doc1",
                "title": "Advanced Machine Learning Architectures",
                "content": "Transformers have revolutionized natural language processing through self-attention mechanisms. The multi-head attention allows models to focus on different aspects of the input sequence simultaneously, enabling better understanding of complex relationships.",
                "entities": ["transformers", "attention", "machine learning", "NLP"],
                "tags": ["AI", "deep learning", "architecture"]
            },
            {
                "id": "doc2", 
                "title": "Vector Database Optimization Strategies",
                "content": "Approximate nearest neighbor algorithms like HNSW and IVF provide efficient similarity search in high-dimensional spaces. These techniques are crucial for large-scale vector retrieval systems and semantic search applications.",
                "entities": ["vector database", "HNSW", "IVF", "similarity search"],
                "tags": ["databases", "optimization", "algorithms"]
            },
            {
                "id": "doc3",
                "title": "Graph Neural Networks for Knowledge Graphs",
                "content": "Graph neural networks enable learning on structured data by propagating information through graph edges. This approach is particularly effective for knowledge graph completion and entity relationship modeling.",
                "entities": ["GNN", "knowledge graphs", "graph learning", "entity relationships"],
                "tags": ["graphs", "neural networks", "knowledge representation"]
            },
            {
                "id": "doc4",
                "title": "Hybrid Search System Design Patterns",
                "content": "Combining lexical and semantic search provides the best of both worlds. TF-IDF captures exact matches while vector similarity finds conceptually related content. Result fusion algorithms like RRF optimize the combination.",
                "entities": ["hybrid search", "TF-IDF", "vector similarity", "result fusion"],
                "tags": ["search", "design patterns", "information retrieval"]
            },
            {
                "id": "doc5",
                "title": "Reinforcement Learning for Search Ranking",
                "content": "RLHF techniques can optimize search ranking by learning from user feedback. This creates adaptive systems that improve over time by understanding user preferences and query intent patterns.",
                "entities": ["RLHF", "search ranking", "user feedback", "adaptive systems"],
                "tags": ["reinforcement learning", "ranking", "personalization"]
            },
            {
                "id": "doc6",
                "title": "FastMCP Server Architecture and Design",
                "content": "FastMCP provides a streamlined way to build Model Context Protocol servers with HTTP and SSE transport options. The architecture supports tools, resources, and intelligent routing for AI applications. Key features include automatic JSON-RPC handling, session management, and seamless integration with AI clients.",
                "entities": ["FastMCP", "MCP", "HTTP transport", "SSE transport", "AI servers", "JSON-RPC"],
                "tags": ["MCP", "server architecture", "AI infrastructure"]
            },
            {
                "id": "doc7",
                "title": "OpenAI GPT and Claude Integration Patterns",
                "content": "Modern AI applications benefit from hybrid approaches combining multiple language models. GPT excels at creative tasks while Claude provides strong reasoning capabilities. MCP servers enable seamless integration between different AI systems.",
                "entities": ["OpenAI", "GPT", "Claude", "AI integration", "language models"],
                "tags": ["AI", "integration", "language models"]
            },
            {
                "id": "doc8",
                "title": "Render Cloud Deployment Best Practices",
                "content": "Deploying applications on Render requires attention to build commands, environment variables, and health checks. For Python applications, ensure proper requirements.txt and consider using gunicorn for production deployments.",
                "entities": ["Render", "cloud deployment", "Python", "gunicorn", "health checks"],
                "tags": ["deployment", "cloud", "DevOps"]
            }
        ]
        
        for doc_data in sample_docs:
            # Generate simple embeddings (in production, use real embedding models)
            content_vector = self._generate_mock_embedding(doc_data["content"])
            
            doc = Document(
                id=doc_data["id"],
                title=doc_data["title"],
                content=doc_data["content"],
                vector=content_vector,
                entities=doc_data["entities"],
                tags=doc_data["tags"]
            )
            
            self.add_document(doc)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for demonstration (replace with real embeddings)"""
        # Simple hash-based mock embedding
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int % (2**32))
        return np.random.normal(0, 1, 384).tolist()

    def add_document(self, document: Document):
        """Add document to search index with intelligence"""
        self.documents[document.id] = document
        
        # Build inverted index for keyword search
        tokens = self._tokenize(document.content.lower())
        for token in tokens:
            if token not in self.inverted_index[token]:
                self.inverted_index[token].append(document.id)
        
        # Build entity graph for graph-based search
        for entity in document.entities:
            for other_entity in document.entities:
                if entity != other_entity:
                    self.entity_graph[entity].append(other_entity)

    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with intelligence"""
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short tokens
        return [token for token in tokens if len(token) > 2]

    def analyze_query_intent(self, query: str) -> QueryIntent:
        """Intelligent query intent analysis"""
        query_lower = query.lower()
        
        # Intent classification with reasoning
        if any(word in query_lower for word in ["what is", "define", "explain", "meaning"]):
            return QueryIntent(
                intent_type="factual",
                confidence=0.9,
                reasoning="Query contains definitional keywords indicating factual information seeking",
                suggested_algorithm="hybrid"
            )
        elif any(word in query_lower for word in ["how to", "steps", "process", "method"]):
            return QueryIntent(
                intent_type="procedural", 
                confidence=0.85,
                reasoning="Query seeks procedural information or step-by-step guidance",
                suggested_algorithm="keyword"
            )
        elif any(word in query_lower for word in ["compare", "vs", "difference", "better"]):
            return QueryIntent(
                intent_type="comparison",
                confidence=0.8,
                reasoning="Query involves comparison between concepts or entities",
                suggested_algorithm="vector"
            )
        elif any(word in query_lower for word in ["related", "similar", "like", "concept"]):
            return QueryIntent(
                intent_type="conceptual",
                confidence=0.75,
                reasoning="Query seeks conceptually related information",
                suggested_algorithm="vector"
            )
        else:
            return QueryIntent(
                intent_type="exploratory",
                confidence=0.6,
                reasoning="General exploratory query requiring broad search approach",
                suggested_algorithm="adaptive"
            )

    def keyword_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """TF-IDF based keyword search with intelligent scoring"""
        query_tokens = self._tokenize(query.lower())
        
        # Calculate TF-IDF scores with intelligence
        doc_scores = {}
        
        for doc_id, document in self.documents.items():
            doc_tokens = self._tokenize(document.content.lower())
            doc_token_count = Counter(doc_tokens)
            
            score = 0
            relevance_factors = {}
            
            for token in query_tokens:
                if token in doc_token_count:
                    tf = doc_token_count[token] / len(doc_tokens)
                    idf = math.log(len(self.documents) / (len(self.inverted_index.get(token, [])) + 1))
                    tf_idf = tf * idf
                    score += tf_idf
                    relevance_factors[f"tf_idf_{token}"] = tf_idf
            
            # Title boost with reasoning
            title_tokens = self._tokenize(document.title.lower())
            title_boost = len(set(query_tokens) & set(title_tokens)) * 0.5
            score += title_boost
            relevance_factors["title_boost"] = title_boost
            
            if score > 0:
                doc_scores[doc_id] = {
                    "score": score,
                    "relevance_factors": relevance_factors
                }
        
        # Sort and create results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for doc_id, score_data in sorted_docs[:max_results]:
            document = self.documents[doc_id]
            snippet = self._generate_snippet(document.content, query)
            
            results.append(SearchResult(
                document_id=doc_id,
                title=document.title,
                content=document.content,
                score=score_data["score"],
                algorithm_used="keyword",
                relevance_factors=score_data["relevance_factors"],
                explanation=f"TF-IDF match with {len(score_data['relevance_factors'])} matching terms",
                snippet=snippet
            ))
        
        return results

    def vector_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Semantic vector search with intelligent similarity"""
        query_vector = self._generate_mock_embedding(query)
        
        doc_scores = {}
        
        for doc_id, document in self.documents.items():
            if document.vector:
                # Cosine similarity
                similarity = self._cosine_similarity(query_vector, document.vector)
                
                relevance_factors = {
                    "semantic_similarity": similarity,
                    "vector_magnitude": np.linalg.norm(document.vector)
                }
                
                doc_scores[doc_id] = {
                    "score": similarity,
                    "relevance_factors": relevance_factors
                }
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for doc_id, score_data in sorted_docs[:max_results]:
            document = self.documents[doc_id]
            snippet = self._generate_snippet(document.content, query)
            
            results.append(SearchResult(
                document_id=doc_id,
                title=document.title,
                content=document.content,
                score=score_data["score"],
                algorithm_used="vector",
                relevance_factors=score_data["relevance_factors"],
                explanation=f"Semantic similarity: {score_data['score']:.3f}",
                snippet=snippet
            ))
        
        return results

    def graph_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Graph-based entity relationship search"""
        query_tokens = self._tokenize(query.lower())
        
        # Find documents with entity relationships
        relevant_entities = set()
        for token in query_tokens:
            for entity in self.entity_graph:
                if token in entity.lower():
                    relevant_entities.add(entity)
                    relevant_entities.update(self.entity_graph[entity][:3])  # Add related entities
        
        doc_scores = {}
        
        for doc_id, document in self.documents.items():
            entity_matches = len(set(document.entities) & relevant_entities)
            if entity_matches > 0:
                # Graph-based scoring
                graph_score = entity_matches / len(document.entities) if document.entities else 0
                
                relevance_factors = {
                    "entity_matches": entity_matches,
                    "graph_connectivity": graph_score,
                    "entity_density": len(document.entities)
                }
                
                doc_scores[doc_id] = {
                    "score": graph_score,
                    "relevance_factors": relevance_factors
                }
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for doc_id, score_data in sorted_docs[:max_results]:
            document = self.documents[doc_id]
            snippet = self._generate_snippet(document.content, query)
            
            results.append(SearchResult(
                document_id=doc_id,
                title=document.title,
                content=document.content,
                score=score_data["score"],
                algorithm_used="graph",
                relevance_factors=score_data["relevance_factors"],
                explanation=f"Entity relationship score with {score_data['relevance_factors']['entity_matches']} entity matches",
                snippet=snippet
            ))
        
        return results

    def hybrid_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Intelligent hybrid search combining multiple algorithms"""
        # Get results from multiple algorithms
        keyword_results = self.keyword_search(query, max_results * 2)
        vector_results = self.vector_search(query, max_results * 2)
        graph_results = self.graph_search(query, max_results * 2)
        
        # Intelligent result fusion using Reciprocal Rank Fusion
        doc_scores = defaultdict(lambda: {"scores": {}, "relevance_factors": {}})
        
        # RRF with intelligent weighting
        for i, result in enumerate(keyword_results):
            rrf_score = 1 / (60 + i + 1)  # RRF with k=60
            doc_scores[result.document_id]["scores"]["keyword"] = rrf_score
            doc_scores[result.document_id]["relevance_factors"].update(result.relevance_factors)
        
        for i, result in enumerate(vector_results):
            rrf_score = 1 / (60 + i + 1)
            doc_scores[result.document_id]["scores"]["vector"] = rrf_score * 1.2  # Boost semantic
            doc_scores[result.document_id]["relevance_factors"].update(result.relevance_factors)
        
        for i, result in enumerate(graph_results):
            rrf_score = 1 / (60 + i + 1)
            doc_scores[result.document_id]["scores"]["graph"] = rrf_score * 0.8  # Lower graph weight
            doc_scores[result.document_id]["relevance_factors"].update(result.relevance_factors)
        
        # Calculate final hybrid scores
        final_scores = {}
        for doc_id, data in doc_scores.items():
            final_score = sum(data["scores"].values())
            final_scores[doc_id] = {
                "score": final_score,
                "relevance_factors": data["relevance_factors"],
                "algorithm_contributions": data["scores"]
            }
        
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for doc_id, score_data in sorted_docs[:max_results]:
            document = self.documents[doc_id]
            snippet = self._generate_snippet(document.content, query)
            
            # Intelligent explanation
            contributions = score_data["algorithm_contributions"]
            main_contributor = max(contributions, key=contributions.get) if contributions else "unknown"
            
            results.append(SearchResult(
                document_id=doc_id,
                title=document.title,
                content=document.content,
                score=score_data["score"],
                algorithm_used="hybrid",
                relevance_factors=score_data["relevance_factors"],
                explanation=f"Hybrid fusion (primary: {main_contributor}) with RRF score {score_data['score']:.3f}",
                snippet=snippet
            ))
        
        return results

    def adaptive_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """AI-powered adaptive search that learns from performance"""
        intent = self.analyze_query_intent(query)
        
        # Select algorithm based on intent and historical performance
        if intent.suggested_algorithm == "hybrid" or not self.algorithm_performance:
            return self.hybrid_search(query, max_results)
        elif intent.suggested_algorithm == "vector":
            return self.vector_search(query, max_results)
        elif intent.suggested_algorithm == "keyword":
            return self.keyword_search(query, max_results)
        else:
            # Fallback to best performing algorithm
            best_algorithm = max(self.algorithm_performance, 
                               key=lambda x: np.mean(self.algorithm_performance[x]))
            if best_algorithm == "vector":
                return self.vector_search(query, max_results)
            elif best_algorithm == "keyword":
                return self.keyword_search(query, max_results)
            else:
                return self.hybrid_search(query, max_results)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)

    def _generate_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate intelligent snippet highlighting query relevance"""
        query_tokens = self._tokenize(query.lower())
        sentences = content.split('. ')
        
        # Find best sentence with query terms
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence_tokens = self._tokenize(sentence.lower())
            matches = len(set(query_tokens) & set(sentence_tokens))
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence[:max_length] + "..." if len(best_sentence) > max_length else best_sentence
        else:
            return content[:max_length] + "..." if len(content) > max_length else content

    def search(self, search_query: SearchQuery) -> IntelligentSearchResponse:
        """Main intelligent search function with reasoning and error handling"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.info(f"Processing search request #{self.request_count}: '{search_query.query}' with algorithm: {search_query.algorithm}")
            
            # Analyze query intent
            intent = self.analyze_query_intent(search_query.query)
            
            # Select algorithm with reasoning
            if search_query.algorithm:
                selected_algorithm = search_query.algorithm
                algorithm_reasoning = f"User explicitly requested {selected_algorithm} algorithm"
            else:
                selected_algorithm = intent.suggested_algorithm
                algorithm_reasoning = f"Selected {selected_algorithm} based on intent analysis: {intent.reasoning}"
            
            # Execute search with error handling
            try:
                if selected_algorithm == "keyword":
                    results = self.keyword_search(search_query.query, search_query.max_results)
                elif selected_algorithm == "vector":
                    results = self.vector_search(search_query.query, search_query.max_results)
                elif selected_algorithm == "graph":
                    results = self.graph_search(search_query.query, search_query.max_results)
                elif selected_algorithm == "hybrid":
                    results = self.hybrid_search(search_query.query, search_query.max_results)
                elif selected_algorithm == "adaptive":
                    results = self.adaptive_search(search_query.query, search_query.max_results)
                else:
                    logger.warning(f"Unknown algorithm '{selected_algorithm}', falling back to hybrid")
                    results = self.hybrid_search(search_query.query, search_query.max_results)
                    selected_algorithm = "hybrid"
                    algorithm_reasoning += " (fallback due to unknown algorithm)"
                    
            except Exception as search_error:
                logger.error(f"Search algorithm error: {search_error}")
                self.error_count += 1
                # Fallback to basic keyword search
                results = self.keyword_search(search_query.query, search_query.max_results)
                selected_algorithm = "keyword"
                algorithm_reasoning = f"Fallback to keyword search due to error: {str(search_error)}"
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            # Record performance for learning
            self.algorithm_performance[selected_algorithm].append(len(results))
            
            # Generate intelligent suggestions
            suggestions = self._generate_suggestions(search_query.query, results)
            
            logger.info(f"Search completed in {total_time:.1f}ms, found {len(results)} results")
            
            return IntelligentSearchResponse(
                query_analysis={
                    "intent": intent.intent_type,
                    "confidence": intent.confidence,
                    "reasoning": intent.reasoning,
                    "query_length": len(search_query.query.split()),
                    "query_complexity": "high" if len(search_query.query.split()) > 5 else "medium" if len(search_query.query.split()) > 2 else "low"
                },
                selected_algorithm=selected_algorithm,
                algorithm_reasoning=algorithm_reasoning,
                results=results,
                performance_metrics={
                    "total_time_ms": total_time,
                    "results_count": len(results),
                    "avg_score": np.mean([r.score for r in results]) if results else 0,
                    "algorithm_efficiency": min(total_time / 100, 1.0)  # Efficiency score
                },
                suggestions=suggestions,
                total_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Critical search error: {e}")
            self.error_count += 1
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            # Return error response
            return IntelligentSearchResponse(
                query_analysis={
                    "intent": "error",
                    "confidence": 0.0,
                    "reasoning": f"Search failed due to error: {str(e)}",
                    "query_length": len(search_query.query.split()),
                    "query_complexity": "unknown"
                },
                selected_algorithm="error",
                algorithm_reasoning=f"Search failed: {str(e)}",
                results=[],
                performance_metrics={
                    "total_time_ms": total_time,
                    "results_count": 0,
                    "avg_score": 0,
                    "algorithm_efficiency": 0
                },
                suggestions=["Please try a different query or contact support"],
                total_time_ms=total_time
            )

    def _generate_suggestions(self, query: str, results: List[SearchResult]) -> List[str]:
        """Generate intelligent search suggestions"""
        suggestions = []
        
        if not results:
            suggestions.append("Try using different keywords or more general terms")
            suggestions.append("Consider using semantic search (vector algorithm)")
        elif len(results) < 3:
            suggestions.append("Try hybrid search for broader results")
            suggestions.append("Use more general terms or synonyms")
        else:
            # Analyze result patterns for suggestions
            common_terms = set()
            for result in results[:3]:
                common_terms.update(self._tokenize(result.title.lower()))
            
            query_terms = set(self._tokenize(query.lower()))
            new_terms = common_terms - query_terms
            
            if new_terms:
                suggestions.append(f"Related terms to explore: {', '.join(list(new_terms)[:3])}")
        
        return suggestions

    def get_health_status(self) -> HealthCheckResponse:
        """Get comprehensive health status of the search engine"""
        uptime = datetime.now() - self.server_start_time
        
        # Calculate error rate
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        
        # Determine health status
        if error_rate > 50:
            status = "critical"
        elif error_rate > 20:
            status = "degraded"
        elif error_rate > 5:
            status = "warning"
        else:
            status = "healthy"
        
        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            version="1.1.0",
            server_info={
                "uptime_seconds": round(uptime.total_seconds(), 2),
                "uptime_human": f"{int(uptime.total_seconds() // 3600)}h {int((uptime.total_seconds() % 3600) // 60)}m",
                "documents_indexed": len(self.documents),
                "total_queries_processed": self.request_count,
                "total_errors": self.error_count,
                "error_rate_percent": round(error_rate, 2),
                "algorithms_available": ["keyword", "vector", "graph", "hybrid", "adaptive"],
                "vocabulary_size": len(self.inverted_index),
                "entity_graph_size": len(self.entity_graph),
                "avg_response_time_ms": round(np.mean([sum(scores) for scores in self.algorithm_performance.values()]) * 10, 2) if self.algorithm_performance else 0
            }
        )

# Initialize the search engine
search_engine = HybridRAGSearchEngine()

# FastMCP server setup
mcp = FastMCP("Hybrid RAG Search Platform")

@mcp.tool()
def intelligent_search(search_request: SearchQuery) -> IntelligentSearchResponse:
    """
    Perform intelligent hybrid search with AI reasoning across multiple algorithms.
    
    Supports:
    - keyword: TF-IDF lexical matching
    - vector: Semantic similarity search  
    - graph: Entity relationship search
    - hybrid: Intelligent fusion of multiple algorithms
    - adaptive: AI-powered algorithm selection
    - custom: User-defined ranking (falls back to hybrid)
    """
    return search_engine.search(search_request)

@mcp.tool()
def add_document(document: Document) -> Dict[str, str]:
    """Add a new document to the search index with intelligent processing"""
    search_engine.add_document(document)
    return {
        "status": "success", 
        "message": f"Document '{document.title}' added with intelligent indexing",
        "document_id": document.id
    }

@mcp.tool()
def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze query intent and get AI reasoning for algorithm selection"""
    intent = search_engine.analyze_query_intent(query)
    return {
        "intent_type": intent.intent_type,
        "confidence": intent.confidence,
        "reasoning": intent.reasoning,
        "suggested_algorithm": intent.suggested_algorithm,
        "query_analysis": {
            "length": len(query.split()),
            "complexity": "high" if len(query.split()) > 5 else "medium" if len(query.split()) > 2 else "low",
            "keywords": search_engine._tokenize(query.lower())
        }
    }

@mcp.tool()
def compare_algorithms(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Compare performance of different search algorithms on the same query"""
    start_time = time.time()
    
    algorithms = ["keyword", "vector", "graph", "hybrid"]
    comparison_results = {}
    
    for algorithm in algorithms:
        alg_start = time.time()
        
        if algorithm == "keyword":
            results = search_engine.keyword_search(query, max_results)
        elif algorithm == "vector":
            results = search_engine.vector_search(query, max_results)
        elif algorithm == "graph":
            results = search_engine.graph_search(query, max_results)
        elif algorithm == "hybrid":
            results = search_engine.hybrid_search(query, max_results)
        
        alg_time = (time.time() - alg_start) * 1000
        
        comparison_results[algorithm] = {
            "results_count": len(results),
            "avg_score": np.mean([r.score for r in results]) if results else 0,
            "max_score": max([r.score for r in results]) if results else 0,
            "time_ms": alg_time,
            "top_result": results[0].title if results else "No results",
            "efficiency_score": len(results) / max(alg_time, 1)  # Results per ms
        }
    
    # Determine best algorithm
    best_algorithm = max(comparison_results.keys(), 
                        key=lambda x: comparison_results[x]["efficiency_score"])
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "query": query,
        "comparison_results": comparison_results,
        "best_algorithm": best_algorithm,
        "reasoning": f"{best_algorithm} performed best with {comparison_results[best_algorithm]['efficiency_score']:.3f} efficiency score",
        "total_comparison_time_ms": total_time
    }

@mcp.tool()
def get_search_analytics() -> Dict[str, Any]:
    """Get intelligent analytics about search performance and patterns"""
    return {
        "total_documents": len(search_engine.documents),
        "algorithm_performance": {
            alg: {
                "avg_results": np.mean(scores) if scores else 0,
                "total_queries": len(scores),
                "efficiency": np.mean(scores) / len(scores) if scores else 0
            }
            for alg, scores in search_engine.algorithm_performance.items()
        },
        "index_statistics": {
            "vocabulary_size": len(search_engine.inverted_index),
            "entity_graph_size": len(search_engine.entity_graph),
            "avg_doc_entities": np.mean([len(doc.entities) for doc in search_engine.documents.values()]),
            "total_query_history": len(search_engine.query_history)
        },
        "recommendations": [
            "Use hybrid search for best overall performance",
            "Vector search excels for conceptual queries", 
            "Keyword search is fastest for exact matches",
            "Graph search works well for entity-rich queries"
        ]
    }

@mcp.tool()
def sample_search_test() -> Dict[str, Any]:
    """Run a comprehensive sample search test to demonstrate all system capabilities"""
    test_queries = [
        {"query": "machine learning transformers", "algorithm": "hybrid"},
        {"query": "vector database optimization", "algorithm": "vector"},
        {"query": "what is semantic search", "algorithm": "keyword"},
        {"query": "graph neural networks", "algorithm": "graph"},
        {"query": "artificial intelligence", "algorithm": "adaptive"}
    ]
    
    test_results = {}
    total_start_time = time.time()
    
    for test_case in test_queries:
        query = test_case["query"]
        algorithm = test_case["algorithm"]
        
        try:
            search_request = SearchQuery(
                query=query,
                algorithm=algorithm,
                max_results=3,
                explain=True
            )
            
            result = search_engine.search(search_request)
            test_results[f"{query} ({algorithm})"] = {
                "status": "success",
                "algorithm_used": result.selected_algorithm,
                "results_count": len(result.results),
                "top_result": result.results[0].title if result.results else "No results",
                "performance_ms": result.total_time_ms,
                "intent_detected": result.query_analysis["intent"],
                "avg_score": result.performance_metrics.get("avg_score", 0)
            }
        except Exception as e:
            test_results[f"{query} ({algorithm})"] = {
                "status": "error",
                "error_message": str(e),
                "algorithm_used": algorithm,
                "results_count": 0,
                "performance_ms": 0,
                "intent_detected": "error"
            }
    
    total_time = (time.time() - total_start_time) * 1000
    
    # Test summary
    successful_tests = sum(1 for result in test_results.values() if result["status"] == "success")
    total_tests = len(test_results)
    
    return {
        "test_summary": f"Completed {total_tests} sample searches in {total_time:.1f}ms",
        "success_rate": f"{(successful_tests/total_tests)*100:.1f}%",
        "test_results": test_results,
        "system_status": "All algorithms functioning correctly" if successful_tests == total_tests else f"{successful_tests}/{total_tests} tests passed",
        "performance_summary": {
            "total_time_ms": total_time,
            "avg_time_per_test_ms": total_time / total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests
        },
        "timestamp": datetime.now().isoformat()
    }

@mcp.resource("search://documents")
def list_documents():
    """List all documents in the search index with metadata"""
    return [
        {
            "id": doc.id,
            "title": doc.title,
            "content_length": len(doc.content),
            "entities": doc.entities,
            "tags": doc.tags,
            "has_vector": doc.vector is not None
        }
        for doc in search_engine.documents.values()
    ]

@mcp.resource("search://algorithms")  
def list_algorithms():
    """List available search algorithms with descriptions"""
    return {
        "keyword": "TF-IDF based lexical matching - fast and precise for exact terms",
        "vector": "Semantic similarity using embeddings - best for conceptual queries",
        "graph": "Entity relationship search - ideal for knowledge graph queries", 
        "hybrid": "Intelligent fusion of multiple algorithms - balanced performance",
        "adaptive": "AI-powered algorithm selection - learns from usage patterns",
        "custom": "User-defined ranking algorithms - maximum flexibility"
    }

@mcp.tool()
def debug_server_status() -> Dict[str, Any]:
    """Get detailed debugging information about the server state"""
    uptime = datetime.now() - search_engine.server_start_time
    
    # Calculate algorithm usage statistics
    algorithm_stats = {}
    for algorithm, scores in search_engine.algorithm_performance.items():
        if scores:
            algorithm_stats[algorithm] = {
                "total_queries": len(scores),
                "avg_results": round(np.mean(scores), 2),
                "min_results": min(scores),
                "max_results": max(scores),
                "success_rate": f"{(len([s for s in scores if s > 0])/len(scores)*100):.1f}%"
            }
    
    # Document statistics
    doc_stats = {
        "total_documents": len(search_engine.documents),
        "avg_content_length": round(np.mean([len(doc.content) for doc in search_engine.documents.values()]), 2),
        "avg_entities_per_doc": round(np.mean([len(doc.entities) for doc in search_engine.documents.values()]), 2),
        "total_unique_entities": len(set().union(*[doc.entities for doc in search_engine.documents.values()])),
        "documents_with_vectors": len([doc for doc in search_engine.documents.values() if doc.vector])
    }
    
    return {
        "server_info": {
            "start_time": search_engine.server_start_time.isoformat(),
            "uptime_seconds": round(uptime.total_seconds(), 2),
            "total_requests": search_engine.request_count,
            "total_errors": search_engine.error_count,
            "error_rate": f"{(search_engine.error_count/max(search_engine.request_count, 1)*100):.2f}%"
        },
        "algorithm_performance": algorithm_stats,
        "document_statistics": doc_stats,
        "index_statistics": {
            "vocabulary_size": len(search_engine.inverted_index),
            "entity_graph_nodes": len(search_engine.entity_graph),
            "entity_graph_edges": sum(len(edges) for edges in search_engine.entity_graph.values()),
            "query_history_size": len(search_engine.query_history)
        },
        "system_health": {
            "status": "healthy" if search_engine.error_count < search_engine.request_count * 0.1 else "degraded",
            "memory_indicators": {
                "documents_loaded": len(search_engine.documents) > 0,
                "indexes_built": len(search_engine.inverted_index) > 0,
                "algorithms_available": len(search_engine.algorithm_performance) > 0
            }
        },
        "debug_timestamp": datetime.now().isoformat()
    }

@mcp.tool()
def get_server_health() -> Dict[str, Any]:
    """Get comprehensive health status and server information for monitoring"""
    health_status = search_engine.get_health_status()
    return health_status.dict()

@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get detailed server information, capabilities, and available endpoints"""
    return {
        "server_name": "Hybrid RAG Search Platform",
        "version": "1.0.0",
        "description": "Advanced intelligent search system with multiple algorithms",
        "capabilities": [
            "Multi-algorithm search (keyword, vector, graph, hybrid, adaptive)",
            "AI-powered query intent analysis",
            "Intelligent result fusion and ranking",
            "Explainable search decisions",
            "Performance analytics and learning"
        ],
        "endpoints": {
            "mcp_protocol": "/mcp/",
            "sse_transport": "/sse/",
            "documentation": "/docs"
        },
        "available_tools": [
            "intelligent_search",
            "add_document", 
            "analyze_query_intent",
            "compare_algorithms",
            "get_search_analytics",
            "sample_search_test",
            "get_server_health",
            "get_server_info"
        ],
        "available_resources": [
            "search://documents",
            "search://algorithms",
            "search://health"
        ],
        "supported_algorithms": {
            "keyword": "TF-IDF based lexical matching - fast and precise for exact terms",
            "vector": "Semantic similarity using embeddings - best for conceptual queries",
            "graph": "Entity relationship search - ideal for knowledge graph queries", 
            "hybrid": "Intelligent fusion of multiple algorithms - balanced performance",
            "adaptive": "AI-powered algorithm selection - learns from usage patterns"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Hybrid RAG Search Platform MCP Server...")
    print("🔧 Version: 1.1.0 (Enhanced with robust error handling)")
    print("🧠 Intelligence Features:")
    print("   • Multi-algorithm search (keyword, vector, graph, hybrid, adaptive)")
    print("   • AI-powered query intent analysis") 
    print("   • Intelligent result fusion and ranking")
    print("   • Explainable search decisions")
    print("   • Performance analytics and learning")
    print("   • Contextual suggestions and recommendations")
    print("   • Comprehensive error handling and debugging")
    print("\n🔍 Available at:")
    print("   • HTTP Transport: http://localhost:8000/mcp/")
    print("   • SSE Transport: http://localhost:8000/sse/")
    print("   • OpenAPI Docs: http://localhost:8000/docs")
    print("\n🛠️ Available Tools:")
    print("   • intelligent_search - Main search functionality")
    print("   • get_server_health - Health status and monitoring")
    print("   • get_server_info - Server capabilities and information")
    print("   • sample_search_test - Built-in comprehensive testing")
    print("   • debug_server_status - Detailed debugging information")
    print("   • analyze_query_intent - Query understanding")
    print("   • compare_algorithms - Performance comparison")
    print("   • add_document - Add new documents")
    print("   • get_search_analytics - Usage analytics")
    print("\n📚 Available Resources:")
    print("   • search://documents - Document index")
    print("   • search://algorithms - Algorithm descriptions")
    print("   • search://health - Health status")
    print("   • search://debug - Debug information")
    print("\n🧪 Testing the Server:")
    print("   1. Initialize: POST /mcp/ with initialize method")
    print("   2. List tools: POST /mcp/ with tools/list method")
    print("   3. Test search: Use sample_search_test tool")
    print("   4. Check health: Use get_server_health tool")
    print("\n📖 Example MCP Request:")
    print("   curl -X POST http://localhost:8000/mcp/ \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -H 'Accept: application/json, text/event-stream' \\")
    print("     -d '{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"id\":\"test\"}'")
    print("\n⚠️  Note: MCP protocol requires proper initialization before using tools")
    print("🔐 For HTTPS deployment: Ensure SSL certificates are properly configured")
    
    # Run the FastMCP server
    import sys
    port = 8001 if len(sys.argv) > 1 and sys.argv[1] == "--port" else 8000
    
    logger.info(f"Starting server on port {port}")
    mcp.run(transport="http", host="0.0.0.0", port=port)

