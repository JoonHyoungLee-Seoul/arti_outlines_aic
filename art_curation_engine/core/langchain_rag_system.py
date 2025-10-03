#!/usr/bin/env python3
"""
LangChain + Chroma RAG System

Advanced RAG implementation using LangChain and Chroma vector database
for better text splitting, embeddings, and retrieval capabilities.

Features:
- RecursiveCharacterTextSplitter for intelligent chunking
- Chroma vector database for persistent storage
- HuggingFace embeddings with caching
- Hybrid search capabilities
- Better metadata handling
"""

import os
import json
import logging
import re
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Set HuggingFace cache to local directory
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
os.environ['HF_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface/hub')

# Ensure cache directories exist
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)

# Load environment variables
load_dotenv()

try:
    # LangChain imports
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    # Try to import FAISS from community package, fallback if unavailable
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("Warning: FAISS not available, will use basic implementation")
        FAISS = None
except ImportError as e:
    print(f"Warning: Some LangChain components unavailable: {e}")
    print("Falling back to basic implementation")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainRAGSystem:
    """
    Advanced RAG system using LangChain components
    
    Features:
    - Intelligent text splitting with RecursiveCharacterTextSplitter
    - HuggingFace embeddings with model caching
    - FAISS vector store for similarity search
    - Rich metadata preservation
    - Better search and ranking
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        self.documents = []
        self.persist_directory = "langchain_vectorstore"
        
        # BM25 components for enhanced fallback search
        self.bm25_index = None
        self.vocabulary = set()
        self.doc_terms = []
        self.idf_scores = {}
        
        self._setup_components()
        
    def _setup_components(self):
        """Initialize LangChain components"""
        logger.info("Setting up LangChain RAG components...")
        
        # Setup embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for stability
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"✅ Embeddings model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
            
        # Setup text splitter with intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly larger chunks for better context
            chunk_overlap=200,  # Good overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting
        )
        logger.info("✅ Text splitter configured")
        
    def load_documents_from_markdown(self, data_dir: str = "data/markdown") -> List[Document]:
        """Load and process documents from markdown files"""
        logger.info(f"Loading documents from {data_dir}...")
        
        data_path = Path(data_dir)
        documents = []
        
        for md_file in data_path.glob("**/*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Create LangChain Document with rich metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "filename": md_file.name,
                        "title": md_file.stem.replace('_', ' ').replace('-', ' '),
                        "file_type": "markdown",
                        "character_count": len(content),
                        "word_count": len(content.split())
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error loading {md_file}: {e}")
                
        logger.info(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def build_vectorstore(self, documents: List[Document] = None):
        """Build vector store from documents"""
        if documents is None:
            documents = self.load_documents_from_markdown()
            
        logger.info("Processing documents with text splitter...")
        
        # Split documents into chunks
        doc_chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
            
        logger.info(f"✅ Created {len(doc_chunks)} document chunks")
        
        # Build vector store
        logger.info("Building vector store...")
        try:
            if FAISS is not None:
                self.vectorstore = FAISS.from_documents(
                    documents=doc_chunks,
                    embedding=self.embeddings
                )
                
                # Save vector store
                self.vectorstore.save_local(self.persist_directory)
                logger.info(f"✅ Vector store built and saved to {self.persist_directory}")
            else:
                # Fallback: store documents in memory for basic search
                self.documents = doc_chunks
                logger.info("✅ Documents stored in memory (FAISS unavailable)")
                # Build BM25 index for enhanced fallback search
                self._build_bm25_index(doc_chunks)
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise
            
        self.documents = doc_chunks
        return len(doc_chunks)
    
    def load_vectorstore(self):
        """Load existing vector store"""
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Vector store loaded from {self.persist_directory}")
                return True
            else:
                logger.info("No existing vector store found")
                return False
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.vectorstore and not self.documents:
            raise ValueError("Vector store or documents not initialized. Call build_vectorstore() first.")
            
        try:
            if self.vectorstore:
                # Use FAISS vector store search
                docs_and_scores = self.vectorstore.similarity_search_with_score(
                    query, k=top_k
                )
                
                results = []
                for doc, score in docs_and_scores:
                    if score >= score_threshold:  # FAISS returns distance, lower is better
                        results.append({
                            'text': doc.page_content,
                            'score': float(1.0 / (1.0 + score)),  # Convert distance to similarity
                            'metadata': doc.metadata,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'title': doc.metadata.get('title', 'Unknown'),
                            'chunk_id': doc.metadata.get('chunk_id', -1)
                        })
                
                return results
            else:
                # Enhanced fallback: BM25 search when available, basic otherwise
                if self.bm25_index:
                    return self._bm25_search(query, top_k)
                else:
                    return self._basic_text_search(query, top_k)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _basic_text_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback basic text search when vector store is unavailable"""
        if not self.documents:
            return []
            
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # Simple scoring based on term frequency
            score = sum(1 for term in query_lower.split() if term in content_lower)
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                'text': doc.page_content,
                'score': float(score / len(query.split())),  # Normalize score
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', 'Unknown'),
                'title': doc.metadata.get('title', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', -1)
            })
        
        return results
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Get LangChain retriever interface"""
        if self.vectorstore:
            search_kwargs = search_kwargs or {"k": 5}
            return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        else:
            # Return a simple retriever-like interface for fallback
            class SimpleRetriever:
                def __init__(self, rag_system, k=5):
                    self.rag_system = rag_system
                    self.k = k
                    
                def get_relevant_documents(self, query: str):
                    results = self.rag_system.search(query, top_k=self.k)
                    # Convert back to Document objects
                    docs = []
                    for result in results:
                        doc = Document(
                            page_content=result['text'],
                            metadata=result['metadata']
                        )
                        docs.append(doc)
                    return docs
            
            search_kwargs = search_kwargs or {"k": 5}
            return SimpleRetriever(self, k=search_kwargs.get("k", 5))
    
    # BM25 Implementation for Enhanced Fallback Search
    
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with stemming-like effects"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
        
        # Simple stemming rules for better matching
        stemmed = []
        for word in words:
            if word.endswith('ing') and len(word) > 4:
                word = word[:-3]
            elif word.endswith('ed') and len(word) > 3:
                word = word[:-2]
            elif word.endswith('s') and len(word) > 3:
                word = word[:-1]
            elif word.endswith('ly') and len(word) > 4:
                word = word[:-2]
            stemmed.append(word)
        
        return stemmed
    
    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 index for enhanced fallback search"""
        logger.info("Building BM25 index for enhanced fallback search...")
        
        # Tokenize all documents
        self.doc_terms = []
        self.vocabulary = set()
        
        for doc in documents:
            terms = self._tokenize(doc.page_content)
            self.doc_terms.append(terms)
            self.vocabulary.update(terms)
        
        # Calculate IDF scores
        self.idf_scores = {}
        total_docs = len(documents)
        
        for term in self.vocabulary:
            doc_count = sum(1 for terms in self.doc_terms if term in terms)
            # Add smoothing to prevent division by zero
            self.idf_scores[term] = math.log((total_docs - doc_count + 0.5) / (doc_count + 0.5))
        
        # Calculate average document length for BM25
        self.avg_doc_length = sum(len(terms) for terms in self.doc_terms) / len(self.doc_terms)
        
        # Mark BM25 as available
        self.bm25_index = True
        
        logger.info(f"✅ BM25 index built: {len(self.vocabulary)} unique terms, avg doc length: {self.avg_doc_length:.1f}")
    
    def _bm25_search(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        """
        BM25 search implementation for enhanced fallback
        
        Args:
            query: Search query
            top_k: Number of results to return
            k1: Controls term frequency effect (1.2-2.0, default 1.5)
            b: Controls length normalization (0-1, default 0.75)
        """
        if not self.bm25_index or not self.documents:
            logger.warning("BM25 index or documents not available")
            return []
            
        query_terms = self._tokenize(query)
        logger.debug(f"BM25 search for query: {query}, terms: {query_terms}")
        scores = []
        matches_found = 0
        
        for i, doc in enumerate(self.documents):
            doc_terms = self.doc_terms[i]
            term_counts = Counter(doc_terms)
            doc_length = len(doc_terms)
            
            # Calculate BM25 score
            score = 0.0
            term_matches = 0
            for term in query_terms:
                if term in term_counts:
                    tf = term_counts[term]  # Term frequency
                    idf = self.idf_scores.get(term, 0)  # Inverse document frequency
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                    term_score = idf * (numerator / denominator)
                    score += term_score
                    term_matches += 1
            
            if score > 0:
                matches_found += 1
                scores.append({
                    'document': doc,
                    'score': score,
                    'index': i,
                    'term_matches': term_matches
                })
        
        logger.debug(f"BM25 found {matches_found} documents with positive scores")
        
        # Sort by relevance score (descending)
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results
        results = []
        for item in scores[:top_k]:
            results.append({
                'text': item['document'].page_content,
                'score': float(item['score']),
                'metadata': item['document'].metadata,
                'source': item['document'].metadata.get('source', 'Unknown'),
                'title': item['document'].metadata.get('title', 'Unknown'),
                'chunk_id': item['document'].metadata.get('chunk_id', -1)
            })
        
        return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search system"""
        stats = {
            'total_documents': len(self.documents) if self.documents else 0,
            'search_method': 'unknown'
        }
        
        if self.vectorstore:
            stats['search_method'] = 'FAISS (semantic)'
        elif self.bm25_index:
            stats['search_method'] = 'BM25 (enhanced)'
            stats['vocabulary_size'] = len(self.vocabulary)
            stats['avg_doc_length'] = self.avg_doc_length
        else:
            stats['search_method'] = 'basic (keyword)'
            
        return stats

def test_langchain_rag():
    """Test the LangChain RAG system"""
    logger.info("Testing LangChain RAG system...")
    
    rag = LangChainRAGSystem()
    
    # Try to load existing vector store
    if not rag.load_vectorstore():
        logger.info("Building new vector store...")
        chunk_count = rag.build_vectorstore()
        logger.info(f"Built vector store with {chunk_count} chunks")
    
    # Display search system statistics
    stats = rag.get_search_stats()
    logger.info(f"\n📊 Search System Stats:")
    logger.info(f"   Method: {stats['search_method']}")
    logger.info(f"   Total Documents: {stats['total_documents']}")
    if 'vocabulary_size' in stats:
        logger.info(f"   Vocabulary Size: {stats['vocabulary_size']}")
        logger.info(f"   Avg Doc Length: {stats['avg_doc_length']:.1f} terms")
    
    # Test searches
    test_queries = [
        "color psychology emotions",
        "red color effects",
        "blue calming effects", 
        "art therapy techniques",
        "stress reduction colors"
    ]
    
    for query in test_queries:
        logger.info(f"\n=== Query: {query} ===")
        results = rag.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. Score: {result['score']:.3f}")
                logger.info(f"   Title: {result['title']}")
                logger.info(f"   Chunk ID: {result['chunk_id']}")
                logger.info(f"   Text: {result['text'][:150]}...")
                logger.info("")
        else:
            logger.info("No relevant results found.")
    
    # Test retriever interface
    logger.info("\n=== Testing Retriever Interface ===")
    retriever = rag.get_retriever({"k": 2})
    docs = retriever.get_relevant_documents("anxiety color therapy")
    logger.info(f"Retriever returned {len(docs)} documents")
    
    # Compare BM25 vs Basic search (if BM25 available)
    if rag.bm25_index:
        logger.info("\n=== BM25 vs Basic Search Comparison ===")
        test_query = "stress anxiety psychology"
        
        bm25_results = rag._bm25_search(test_query, top_k=3)
        basic_results = rag._basic_text_search(test_query, top_k=3)
        
        logger.info(f"Query: '{test_query}'")
        logger.info("BM25 Results:")
        for i, result in enumerate(bm25_results, 1):
            logger.info(f"  {i}. Score: {result['score']:.3f} - {result['title']}")
        
        logger.info("Basic Results:")
        for i, result in enumerate(basic_results, 1):
            logger.info(f"  {i}. Score: {result['score']:.3f} - {result['title']}")
    
    logger.info("✅ LangChain RAG system test completed!")

if __name__ == "__main__":
    test_langchain_rag()