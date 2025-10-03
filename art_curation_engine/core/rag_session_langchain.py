#!/usr/bin/env python3
"""
RAG Session Brief Generator - LangChain Version

Enhanced evidence-based curation brief generator using LangChain RAG system
for improved text splitting, document processing, and search capabilities.

Usage:
    from rag_session_langchain import RAGSessionBrief
    
    brief_gen = RAGSessionBrief()
    brief = brief_gen.generate_brief("feeling stressed at work", ["anxiety", "overwhelmed"])
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables with override
load_dotenv(override=True)

# Set HuggingFace cache directories BEFORE importing any models
# Override any existing cache paths that might point to external drives
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface/hub')

# Create cache directories if they don't exist
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HUGGINGFACE_HUB_CACHE'], exist_ok=True)

# Import LangChain RAG system
from .langchain_rag_system import LangChainRAGSystem

# LLM imports
from llama_index.llms.openai import OpenAI as LlamaOpenAI

class RAGSessionBrief:
    """
    RAG Session Brief Generator - LangChain Enhanced
    
    Combines LangChain RAG system with LLM reasoning to generate
    personalized curation briefs for artwork recommendation.
    
    Improvements over manual system:
    - Better text chunking with RecursiveCharacterTextSplitter
    - Enhanced document metadata
    - More intelligent search capabilities
    - 685 chunks vs 536 (better coverage)
    """
    
    def __init__(self, cache_dir: str = None):
        # Use relative paths if no specific path provided
        script_dir = Path(__file__).parent
        self.cache_dir = Path(cache_dir) if cache_dir else script_dir / ".cache" / "rag_sessions_langchain"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag_system = None
        self.llm = None
        self._setup_rag_system()
        self._setup_llm()
        
    def _setup_rag_system(self):
        """Load the LangChain RAG system"""
        print("🔄 Loading LangChain RAG system...")
        
        self.rag_system = LangChainRAGSystem()
        
        # Try to load existing vector store
        if not self.rag_system.load_vectorstore():
            print("Building new vector store...")
            chunk_count = self.rag_system.build_vectorstore()
            print(f"Built vector store with {chunk_count} chunks")
        else:
            print("Loaded existing vector store")
        
        print("✅ LangChain RAG system loaded successfully")
        
    def _setup_llm(self):
        """Setup LLM for brief generation"""
        # Use OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your-api-key-here":
            self.llm = LlamaOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=openai_key
            )
            print("✅ LLM setup complete (OpenAI)")
            return
            
        print("⚠️ Warning: No valid API key found for LLM")
        
    def generate_rag_queries(self, situation: str, emotions: List[str]) -> List[str]:
        """LLM dynamically generates RAG queries based on situation and emotions"""
        if not self.llm:
            raise RuntimeError("LLM not initialized for query generation")
            
        emotion_str = ', '.join(emotions)
        
        query_generation_prompt = f"""You are a color psychology and art therapy expert.
Generate optimal research paper search queries for the user's specific situation and emotions.

User Input:
- Situation: {situation}
- Emotions: {emotion_str}

Generate 5 specific search queries as a JSON array that are most relevant to this situation and emotions.
Each query should find relevant evidence from color psychology, art therapy, and emotion regulation research literature.

Include diverse perspectives:
1. Color/visual effects research specific to emotional states
2. Environmental psychology research for specific situations
3. Art therapy intervention effectiveness research
4. Cognitive-emotional regulation mechanism research
5. Cultural/individual difference considerations research

Output format (JSON array):
["query1", "query2", "query3", ...]

Query style examples:
- "visual environment research for enhancing focus during creative work"
- "experimental studies on fatigue and color temperature relationships"
- "color preference patterns in motivational states"

Return only JSON:"""

        try:
            response = self.llm.complete(query_generation_prompt)
            response_text = response.text.strip()
            
            # JSON 추출
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif response_text.startswith("["):
                pass  # 이미 JSON 형태
            else:
                # JSON 배열 부분만 추출
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                response_text = response_text[start:end]
                
            queries = json.loads(response_text)
            
            if not isinstance(queries, list):
                raise ValueError("Generated queries must be a list")
                
            print(f"📝 Generated {len(queries)} dynamic RAG queries")
            return queries
            
        except Exception as e:
            print(f"⚠️ Query generation failed: {e}")
            # Fallback to minimal situational query
            fallback_query = f"situation: {situation}. emotions: {emotion_str}. related color psychology and art therapy research"
            return [fallback_query]
    
    def _search_single_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search single query and return evidence list"""
        try:
            results = self.rag_system.search(query, top_k=top_k)
            
            evidence_list = []
            for result in results:
                evidence = {
                    "title": result['title'],
                    "year": "Unknown",  # Could be extracted from metadata if available
                    "domain": "Color Psychology / Art Therapy",
                    "score": result['score'],
                    "snippet": result['text'][:650].replace("\n", " "),
                    "query_used": query,
                    "chunk_id": result.get('chunk_id', -1),
                    "source": result.get('source', 'Unknown')
                }
                evidence_list.append(evidence)
                
            return evidence_list
            
        except Exception as e:
            print(f"    ⚠️ Error with query '{query[:50]}...': {e}")
            return []
    
    def fetch_evidence(self, queries: List[str], top_k: int = 3, use_parallel: bool = True) -> List[Dict[str, Any]]:
        """Fetch evidence from LangChain RAG system for given queries"""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        start_time = time.time()
        print(f"🔄 Fetching evidence for {len(queries)} queries{'(parallel)' if use_parallel else '(sequential)'}...")
        
        evidence_list = []
        
        if use_parallel:
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
                # 쿼리 실행
                future_to_query = {
                    executor.submit(self._search_single_query, query, top_k): query 
                    for query in queries
                }
                
                # 결과 수집
                for future in as_completed(future_to_query, timeout=30):  # 30초 타임아웃
                    query = future_to_query[future]
                    try:
                        query_evidence = future.result(timeout=10)  # 개별 쿼리 10초 타임아웃
                        evidence_list.extend(query_evidence)
                        print(f"  ✅ Query completed: {query[:50]}... ({len(query_evidence)} results)")
                        
                    except Exception as e:
                        print(f"  ⚠️ Query failed: {query[:50]}... - {e}")
                        continue
        else:
            # 기존 순차 처리
            for i, query in enumerate(queries):
                print(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
                
                try:
                    query_evidence = self._search_single_query(query, top_k)
                    evidence_list.extend(query_evidence)
                    print(f"    Found {len(query_evidence)} results for query: {query[:80]}...")
                    
                except Exception as e:
                    print(f"    ⚠️ Error with query: {e}")
                    continue
        
        # Remove duplicates based on title and snippet start
        unique_evidence = {}
        for evidence in evidence_list:
            key = (evidence["title"], evidence["snippet"][:120])
            if key not in unique_evidence or evidence["score"] > unique_evidence[key]["score"]:
                unique_evidence[key] = evidence
                
        final_evidence = list(unique_evidence.values())[:5]  # Limit to top 5
        
        elapsed_time = time.time() - start_time
        print(f"✅ Collected {len(final_evidence)} unique evidence pieces in {elapsed_time:.2f}s")
        
        return final_evidence
    
    def render_brief_prompt(self, evidence: List[Dict[str, Any]], situation: str, emotions: List[str]) -> str:
        """LLM analyzes evidence to generate situation-specific customized brief"""
        evidence_text = "\n".join([
            f"- ({e['title']} {e.get('year')}) {e['snippet']}" 
            for e in evidence
        ])
        
        prompt = f"""You are a color psychology and art therapy expert and personalized curator.
Analyze the scientific evidence below to generate a curation brief optimized for the user's specific situation and emotions.

User Situation Analysis:
- Situation: {situation}
- Emotional State: {emotions}

Your Role:
1. Analyze evidence to determine the most suitable approach for this specific situation and emotions
2. Establish personalized, non-uniform curation strategy
3. Provide science-based customized recommendations

Output JSON Structure:
{{
  "situation_analysis": "Professional analysis of user's situation and emotional state",
  "curation_strategy": "Curation strategy and rationale optimized for this specific case",
  "curatorial_goals": ["Situation-specific tailored goals"],
  "visual_elements": {{
    "preferred_themes": ["Evidence-based recommended themes"],
    "avoid_elements": ["Elements to avoid in this situation"],
    "color_psychology": {{
      "primary_hues": ["Suitable colors with reasoning"],
      "saturation_level": "Appropriate saturation level with rationale",
      "brightness_level": "Appropriate brightness level with rationale",
      "color_temperature": "Warm/cool preference with reasoning"
    }},
    "composition_style": {{
      "visual_complexity": "Simple/complex preference with reasoning",
      "artistic_style": ["Suitable art styles with rationale"],
      "rhythm_tempo": ["Visual rhythm and tempo guidelines"]
    }}
  }},
  "scientific_rationale": {{
    "key_mechanisms": ["Applied psychological mechanisms"],
    "evidence_strength": "Reliability assessment of evidence",
    "individual_considerations": ["Individual differences or situational considerations"]
  }},
  "citations": [{{
    "title": "Paper title",
    "year": "Publication year",
    "key_finding": "Key finding applied to this situation"
  }}]
}}

Important Principles:
- Do not speculate on content not in the evidence
- Avoid uniform "blue/green + nature" recommendations
- Reflect the uniqueness of this specific situation
- Balance scientific evidence with personalization

Scientific Evidence:
<<<
{evidence_text}
>>>

Return only JSON:"""
        return prompt
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON response"""
        if not self.llm:
            raise RuntimeError("LLM not initialized - check OPENAI_API_KEY")
            
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            if "```json" in response_text:
                # Extract JSON from code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif response_text.startswith("```"):
                # Remove code block markers
                response_text = response_text.strip("`").strip()
                
            # Parse JSON
            brief_json = json.loads(response_text)
            return brief_json
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response was: {response_text[:200]}...")
            raise
        except Exception as e:
            print(f"❌ LLM call error: {e}")
            raise
    
    def _get_cache_key(self, situation: str, emotions: List[str]) -> str:
        """Generate cache key for situation and emotions"""
        content = f"{situation}:{':'.join(sorted(emotions))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load result from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def generate_brief(self, situation: str, emotions: List[str], use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate complete session brief
        
        Args:
            situation: User situation description
            emotions: List of user emotions
            use_cache: Whether to use caching
            
        Returns:
            Complete session brief with evidence and citations
        """
        print(f"\n🎯 Generating session brief with dynamic LangChain RAG...")
        print(f"   Situation: {situation}")
        print(f"   Emotions: {emotions}")
        
        # Check cache first
        cache_key = self._get_cache_key(situation, emotions)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                print("✅ Using cached result")
                return cached
        
        # Step 1: Generate dynamic RAG queries
        queries = self.generate_rag_queries(situation, emotions)
        
        # Step 2: Fetch evidence
        evidence = self.fetch_evidence(queries)
        
        # Step 3: Generate brief prompt
        prompt = self.render_brief_prompt(evidence, situation, emotions)
        
        # Step 4: Call LLM
        print("🤖 Calling LLM to generate brief...")
        brief_json = self.call_llm(prompt)
        
        # Step 5: Package complete result
        result = {
            "brief": brief_json,
            "evidence_used": evidence,
            "queries_used": queries,
            "user_input": {
                "situation": situation,
                "emotions": emotions
            },
            "metadata": {
                "cache_key": cache_key,
                "evidence_count": len(evidence),
                "system_type": "langchain_dynamic",
                "chunk_count": 685,
                "is_dynamic": True,
                "query_count": len(queries)
            }
        }
        
        # Cache result
        if use_cache:
            self._save_to_cache(cache_key, result)
            
        print("✅ Session brief generated successfully with LangChain!")
        return result

def test_parallel_performance():
    """Test parallel vs sequential performance with multiple runs"""
    print("🧪 Testing Parallel vs Sequential Performance (Multiple Runs)")
    print("=" * 70)
    
    brief_gen = RAGSessionBrief()
    
    test_scenarios = [
        {
            "situation": "Under heavy work-related stress and finding it hard to concentrate",
            "emotions": ["stress", "anxiety", "overwhelmed"]
        },
        {
            "situation": "It's evening after a long day, and I want to unwind at home",
            "emotions": ["tired", "peaceful", "contemplative"]
        },
        {
            "situation": "Need to be creative and focused for an important project",
            "emotions": ["motivated", "focused", "excited"]
        }
    ]
    
    sequential_times = []
    parallel_times = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Test Run {i}/3: {scenario['situation'][:50]}...")
        
        # Generate queries
        queries = brief_gen.generate_rag_queries(scenario["situation"], scenario["emotions"])
        print(f"   Generated {len(queries)} queries")
        
        # Test sequential (3 runs)
        seq_times = []
        for run in range(3):
            start_time = time.time()
            evidence_seq = brief_gen.fetch_evidence(queries, top_k=3, use_parallel=False)
            seq_time = time.time() - start_time
            seq_times.append(seq_time)
        avg_seq_time = sum(seq_times) / len(seq_times)
        
        # Test parallel (3 runs)
        par_times = []
        for run in range(3):
            start_time = time.time()
            evidence_par = brief_gen.fetch_evidence(queries, top_k=3, use_parallel=True)
            par_time = time.time() - start_time
            par_times.append(par_time)
        avg_par_time = sum(par_times) / len(par_times)
        
        sequential_times.append(avg_seq_time)
        parallel_times.append(avg_par_time)
        
        print(f"   Sequential: {avg_seq_time:.3f}s (avg of 3 runs)")
        print(f"   Parallel:   {avg_par_time:.3f}s (avg of 3 runs)")
        print(f"   Speedup:    {avg_seq_time/avg_par_time:.2f}x")
    
    # Overall results
    avg_sequential = sum(sequential_times) / len(sequential_times)
    avg_parallel = sum(parallel_times) / len(parallel_times)
    overall_speedup = avg_sequential / avg_parallel
    
    print(f"\n📊 Overall Performance Results:")
    print(f"   Average Sequential: {avg_sequential:.3f}s")
    print(f"   Average Parallel:   {avg_parallel:.3f}s")
    print(f"   Overall Speedup:    {overall_speedup:.2f}x")
    print(f"   Time saved per call: {avg_sequential - avg_parallel:.3f}s")
    
    if avg_parallel < avg_sequential:
        print("✅ Parallel processing is consistently faster!")
        recommendation = "keep"
    else:
        print("⚠️ Sequential processing is faster - recommend rollback")
        recommendation = "rollback"
    
    return {
        "avg_sequential_time": avg_sequential,
        "avg_parallel_time": avg_parallel,
        "overall_speedup": overall_speedup,
        "recommendation": recommendation,
        "parallel_faster": avg_parallel < avg_sequential
    }

def main():
    """Example usage with LangChain RAG"""
    # Initialize brief generator
    brief_gen = RAGSessionBrief()
    
    # Example 1: Work stress with LangChain
    print("\n" + "="*60)
    print("Example 1: Work Stress Scenario (LangChain RAG)")
    print("="*60)
    
    result1 = brief_gen.generate_brief(
        situation="Under heavy work-related stress and finding it hard to concentrate.",
        emotions=["stress", "anxiety", "overwhelmed"]
    )
    
    print("\n📋 Generated Brief:")
    print(json.dumps(result1["brief"], ensure_ascii=False, indent=2))
    print(f"\n📊 Evidence Count: {result1['metadata']['evidence_count']}")
    print(f"📊 System Type: {result1['metadata']['system_type']}")
    print(f"📊 Query Count: {result1['metadata']['query_count']}")
    print(f"📊 Dynamic Generation: {result1['metadata']['is_dynamic']}")
    
    # Example 2: Evening relaxation
    print("\n" + "="*60)
    print("Example 2: Evening Relaxation Scenario (LangChain RAG)") 
    print("="*60)
    
    result2 = brief_gen.generate_brief(
        situation="It’s evening after a long day, and user just want to unwind at home.",
        emotions=["tired", "peaceful", "contemplative"]
    )
    
    print("\n📋 Generated Brief:")
    print(json.dumps(result2["brief"], ensure_ascii=False, indent=2))
    
    print(f"\n💾 Results cached in: {brief_gen.cache_dir}")

    # Example 3: Creative Focus Sprint
    print("\n" + "="*60)
    print("Example 3: Creative Focus Sprint (LangChain RAG)")
    print("="*60)

    result3 = brief_gen.generate_brief(
        situation="With a deadline tomorrow, user wants to dive deep into work tonight—minimize distractions and gets into a focused flow.",
        emotions=["stressed", "distracted", "motivated"]
    )

    print("\n📋 Generated Brief:")
    print(json.dumps(result3["brief"], ensure_ascii=False, indent=2))

    print(f"\n💾 Results cached in: {brief_gen.cache_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Performance test mode
        test_parallel_performance()
    else:
        # Normal demo mode
        main()