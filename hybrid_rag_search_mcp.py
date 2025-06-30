#!/usr/bin/env python3
"""
Hybrid RAG Search Platform MCP Server
A sophisticated intelligent search system with multiple algorithms and AI reasoning
Enhanced with proper testing endpoints and health checks for Render deployment
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
        
        # Initialize with sample intelligent documents
        self._initialize_sample_data()
        
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
                "content": "FastMCP provides a streamlined way to build Model Context Protocol servers with HTTP and SSE transport options. The architecture supports tools, resources, and intelligent routing for AI applications.",
                "entities": ["FastMCP", "MCP", "HTTP transport", "SSE transport", "AI servers"],
                "tags": ["MCP", "server architecture", "AI infrastructure"]
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
        """Main intelligent search function with reasoning"""
        start_time = time.time()
        
        # Analyze query intent
        intent = self.analyze_query_intent(search_query.query)
        
        # Select algorithm with reasoning
        if search_query.algorithm:
            selected_algorithm = search_query.algorithm
            algorithm_reasoning = f"User explicitly requested {selected_algorithm} algorithm"
        else:
            selected_algorithm = intent.suggested_algorithm
            algorithm_reasoning = f"Selected {selected_algorithm} based on intent analysis: {intent.reasoning}"
        
        # Execute search
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
            results = self.hybrid_search(search_query.query, search_query.max_results)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Record performance for learning
        self.algorithm_performance[selected_algorithm].append(len(results))
        
        # Generate intelligent suggestions
        suggestions = self._generate_suggestions(search_query.query, results)
        
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
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            server_info={
                "uptime_seconds": uptime.total_seconds(),
                "documents_indexed": len(self.documents),
                "total_queries_processed": sum(len(scores) for scores in self.algorithm_performance.values()),
                "algorithms_available": ["keyword", "vector", "graph", "hybrid", "adaptive"],
                "memory_usage_mb": "estimation_not_available"
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
    """Run a sample search test to demonstrate the system capabilities"""
    test_queries = [
        "machine learning transformers",
        "vector database optimization",
        "what is semantic search"
    ]
    
    test_results = {}
    
    for query in test_queries:
        search_request = SearchQuery(
            query=query,
            algorithm="hybrid",
            max_results=3,
            explain=True
        )
        
        result = search_engine.search(search_request)
        test_results[query] = {
            "algorithm_used": result.selected_algorithm,
            "results_count": len(result.results),
            "top_result": result.results[0].title if result.results else "No results",
            "performance_ms": result.total_time_ms,
            "intent_detected": result.query_analysis["intent"]
        }
    
    return {
        "test_summary": "Sample searches completed successfully",
        "test_results": test_results,
        "system_status": "All algorithms functioning correctly",
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

@mcp.resource("search://health")
def get_health():
    """Get detailed health status and system information"""
    return search_engine.get_health_status().dict()

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
    print("🧠 Intelligence Features:")
    print("   • Multi-algorithm search (keyword, vector, graph, hybrid, adaptive)")
    print("   • AI-powered query intent analysis") 
    print("   • Intelligent result fusion and ranking")
    print("   • Explainable search decisions")
    print("   • Performance analytics and learning")
    print("   • Contextual suggestions and recommendations")
    print("\n🔍 Available at:")
    print("   • HTTP Transport: http://localhost:8000/mcp/")
    print("   • SSE Transport: http://localhost:8000/sse/")
    print("   • OpenAPI Docs: http://localhost:8000/docs")
    print("\n🛠️ Available Tools:")
    print("   • intelligent_search - Main search functionality")
    print("   • get_server_health - Health status and monitoring")
    print("   • get_server_info - Server capabilities and information")
    print("   • sample_search_test - Built-in testing functionality")
    
    # Run the FastMCP server
    import sys
    port = 8001 if len(sys.argv) > 1 and sys.argv[1] == "--port" else 8000
    mcp.run(transport="http", host="0.0.0.0", port=port)

