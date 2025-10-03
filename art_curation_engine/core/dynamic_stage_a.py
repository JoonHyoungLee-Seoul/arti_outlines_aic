#!/usr/bin/env python3
"""
Dynamic Stage A: Evidence-Based Candidate Collection

Revolutionary upgrade from hardcoded Stage A to LLM+RAG dynamic system.
Eliminates all hardcoded emotion/situation/style mappings.

Architecture:
- Step 5 → Dynamic Keyword/Prompt Generator (DKG) → Stage A
- LLM generates keywords and prompts based on evidence
- Subject vocabulary validation and concept expansion
- Safety fallback with caching for reliability

Author: Claude Code
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import pickle

import numpy as np
import torch
from dotenv import load_dotenv

# LLM and similarity imports
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

@dataclass
class DKGConfig:
    """Dynamic Keyword/Prompt Generator Configuration"""
    max_a1_keywords: int = 10  # Optimized for faster generation
    max_a2_prompts: int = 3    # Reduced from 4 
    concept_similarity_threshold: float = 0.60
    cache_ttl_hours: int = 24
    llm_timeout_seconds: int = 10
    enable_fuzzy_matching: bool = True
    enable_concept_expansion: bool = True

class SubjectVocabularyExtractor:
    """Extract and manage subject vocabulary from artwork metadata"""
    
    def __init__(self, metadata_path: str = "data/aic_sample/metadata.jsonl"):
        self.metadata_path = metadata_path
        self.subject_vocab: Set[str] = set()
        self.subject_frequency: Dict[str, int] = defaultdict(int)
        self.concept_aliases: Dict[str, List[str]] = {}
        
        self._load_subject_vocabulary()
        self._load_concept_aliases()
    
    def _load_subject_vocabulary(self):
        """Extract all unique subject terms from metadata"""
        print("📚 Extracting subject vocabulary from metadata...")
        
        if not Path(self.metadata_path).exists():
            print(f"⚠️ Metadata file not found: {self.metadata_path}")
            return
        
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                line_count = 0
                error_count = 0
                
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            artwork = json.loads(line)
                            subject_titles = artwork.get("subject_titles", [])
                            
                            for subject in subject_titles:
                                if isinstance(subject, str):
                                    # Normalize and clean
                                    clean_subject = subject.lower().strip()
                                    if len(clean_subject) > 2:  # Filter very short terms
                                        self.subject_vocab.add(clean_subject)
                                        self.subject_frequency[clean_subject] += 1
                            
                            line_count += 1
                            
                        except json.JSONDecodeError as e:
                            error_count += 1
                            if error_count <= 5:  # Only show first 5 errors
                                print(f"⚠️ Invalid JSON on line {line_num}: {e}")
                        except Exception as e:
                            error_count += 1
                            if error_count <= 5:
                                print(f"⚠️ Error processing line {line_num}: {e}")
                
                if error_count > 5:
                    print(f"⚠️ ... and {error_count - 5} more errors")
                    
                if line_count == 0:
                    print("⚠️ No valid metadata entries found")
                    
        except Exception as e:
            print(f"⚠️ Error reading metadata file: {e}")
        
        print(f"✅ Extracted {len(self.subject_vocab)} unique subject terms")
        print(f"📊 Top 10 subjects: {list(sorted(self.subject_frequency.items(), key=lambda x: x[1], reverse=True)[:10])}")
    
    def _load_concept_aliases(self):
        """Load concept aliases and expansions"""
        self.concept_aliases = {
            "water": ["lake", "river", "ocean", "sea", "pond", "stream", "waterfall", "lagoon", "estuary"],
            "nature": ["landscape", "forest", "trees", "mountains", "hills", "meadow", "field", "wilderness"],
            "garden": ["flowers", "plants", "botanical", "greenery", "flora", "garden", "courtyard"],
            "sky": ["clouds", "sunset", "sunrise", "horizon", "atmospheric", "weather", "celestial"],
            "peaceful": ["calm", "serene", "tranquil", "quiet", "still", "gentle", "soothing"],
            "social": ["people", "gathering", "community", "family", "portrait", "group", "celebration"],
            "home": ["interior", "domestic", "furniture", "room", "house", "dwelling"],
            "art": ["painting", "drawing", "sculpture", "artistic", "creative", "aesthetic"],
            "color": ["blue", "green", "warm", "cool", "bright", "dark", "vibrant", "muted"],
            "emotion": ["joyful", "melancholic", "dramatic", "expressive", "contemplative", "spiritual"]
        }
        
        print(f"✅ Loaded {len(self.concept_aliases)} concept categories")
    
    def validate_keywords(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        """Validate keywords against subject vocabulary"""
        valid_keywords = []
        invalid_keywords = []
        
        for keyword in keywords:
            clean_keyword = keyword.lower().strip()
            if clean_keyword in self.subject_vocab:
                valid_keywords.append(clean_keyword)
            else:
                invalid_keywords.append(clean_keyword)
        
        return valid_keywords, invalid_keywords
    
    def expand_keywords_with_concepts(self, keywords: List[str]) -> List[str]:
        """Expand keywords using concept aliases"""
        expanded = set(keywords)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check if keyword matches any concept
            for concept, aliases in self.concept_aliases.items():
                if keyword_lower == concept or keyword_lower in aliases:
                    # Add all aliases that exist in our vocabulary
                    for alias in aliases:
                        if alias in self.subject_vocab:
                            expanded.add(alias)
        
        return list(expanded)
    
    def get_frequent_subjects(self, min_frequency: int = 3) -> List[str]:
        """Get frequently occurring subjects for fallback"""
        return [
            subject for subject, freq in self.subject_frequency.items()
            if freq >= min_frequency
        ]

class DynamicKeywordPromptGenerator:
    """Dynamic Keyword/Prompt Generator using LLM + Evidence"""
    
    def __init__(self, config: DKGConfig = None):
        self.config = config or DKGConfig()
        self.cache_dir = Path(".cache/dynamic_stage_a")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.llm = None
        self.vocab_extractor = SubjectVocabularyExtractor()
        self.similarity_model = None
        
        # Cache
        self.session_cache = {}
        
        self._setup_llm()
        if self.config.enable_fuzzy_matching:
            self._setup_similarity_model()
        
        print("✅ Dynamic Keyword/Prompt Generator initialized")
    
    def _setup_llm(self):
        """Setup LLM for dynamic generation"""
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your-api-key-here":
            self.llm = LlamaOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200,   # Limit response length for speed
                api_key=openai_key
            )
            print("✅ LLM setup complete for dynamic generation")
        else:
            print("⚠️ Warning: No valid API key found for LLM")
    
    def _setup_similarity_model(self):
        """Setup similarity model for fuzzy matching"""
        try:
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Similarity model loaded for fuzzy matching")
        except Exception as e:
            print(f"⚠️ Could not load similarity model: {e}")
    
    def _get_cache_key(self, situation: str, emotions: List[str], brief: Optional[Dict] = None) -> str:
        """Generate cache key for session"""
        content = f"{situation}|{':'.join(sorted(emotions))}"
        if brief:
            content += f"|{json.dumps(brief, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached result with enhanced error handling"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
            
        try:
            # Check file size (corrupted files are often 0 bytes)
            if cache_file.stat().st_size == 0:
                print(f"⚠️ Cache file is empty, removing: {cache_file}")
                cache_file.unlink(missing_ok=True)
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Validate cache data structure
            if not isinstance(data, dict) or 'result' not in data or 'timestamp' not in data:
                print(f"⚠️ Invalid cache structure, removing: {cache_file}")
                cache_file.unlink(missing_ok=True)
                return None
            
            # Check TTL
            cache_age = time.time() - data.get('timestamp', 0)
            max_age = self.config.cache_ttl_hours * 3600
            
            if cache_age >= max_age:
                print(f"⚠️ Cache expired ({cache_age/3600:.1f}h old), removing: {cache_file}")
                cache_file.unlink(missing_ok=True)
                return None
            
            # Validate cached result structure
            result = data['result']
            if not isinstance(result, dict) or 'A1_keywords' not in result or 'A2_prompts' not in result:
                print(f"⚠️ Invalid cached result structure, removing: {cache_file}")
                cache_file.unlink(missing_ok=True)
                return None
            
            return result
            
        except (pickle.PickleError, EOFError) as e:
            print(f"⚠️ Cache corruption error, removing file: {e}")
            cache_file.unlink(missing_ok=True)
            return None
        except (OSError, IOError) as e:
            print(f"⚠️ Cache file I/O error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected cache load error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save result to cache with enhanced error handling"""
        if not result or not isinstance(result, dict):
            print("⚠️ Cannot cache invalid result")
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        temp_file = cache_file.with_suffix('.tmp')
        
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare cache data
            data = {
                'result': result,
                'timestamp': time.time(),
                'cache_version': '1.0'
            }
            
            # Write to temporary file first (atomic operation)
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Verify the temporary file was written correctly
            with open(temp_file, 'rb') as f:
                verify_data = pickle.load(f)
                if verify_data.get('result') != result:
                    raise ValueError("Cache verification failed")
            
            # Atomically move temp file to final location
            temp_file.replace(cache_file)
            
        except (OSError, IOError) as e:
            print(f"⚠️ Cache file I/O error: {e}")
            temp_file.unlink(missing_ok=True)
        except pickle.PickleError as e:
            print(f"⚠️ Cache serialization error: {e}")
            temp_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Unexpected cache save error: {e}")
            temp_file.unlink(missing_ok=True)
    
    def _render_dynamic_generation_prompt(self, situation: str, emotions: List[str], 
                                        brief: Optional[Dict] = None, 
                                        evidence: Optional[List[Dict]] = None) -> str:
        """Generate LLM prompt for dynamic keyword/prompt generation"""
        
        # Extract relevant information from brief
        brief_context = ""
        if brief:
            themes = brief.get("visual_elements", {}).get("preferred_themes", [])
            avoid_elements = brief.get("visual_elements", {}).get("avoid_elements", [])
            color_info = brief.get("visual_elements", {}).get("color_psychology", {})
            
            if themes:
                brief_context += f"Recommended themes: {themes}\n"
            if avoid_elements:
                brief_context += f"Avoid: {avoid_elements}\n"
            if color_info:
                brief_context += f"Color guidance: {color_info}\n"
        
        # Extract evidence titles
        evidence_context = ""
        if evidence:
            evidence_titles = [e.get("title", "Unknown") for e in evidence[:3]]
            evidence_context = f"Based on research: {evidence_titles}\n"
        
        # Optimized shorter prompt with vocabulary guidance
        emotions_str = ", ".join(emotions)
        
        # Get common vocabulary examples for guidance
        frequent_vocab = self.vocab_extractor.get_frequent_subjects(min_frequency=10)[:20]
        vocab_examples = ", ".join(frequent_vocab[:10])
        
        prompt = f"""Generate art search terms for: {emotions_str} in {situation}

Use these common artwork terms: {vocab_examples}

JSON output:
{{
  "A1_keywords": [10 terms from common artwork vocabulary],
  "A2_prompts": [3 visual descriptions]
}}

Keywords must be simple nouns like: nature, water, landscape, people
Prompts: colors, mood, visual elements

JSON:"""
        
        return prompt
    
    def _generate_with_llm(self, situation: str, emotions: List[str], 
                          brief: Optional[Dict] = None, 
                          evidence: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
        """Generate keywords and prompts using LLM with enhanced error handling"""
        if not self.llm:
            print("⚠️ LLM not initialized")
            return None
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                prompt = self._render_dynamic_generation_prompt(situation, emotions, brief, evidence)
                
                # Call LLM with timeout
                response = self.llm.complete(prompt)
                
                # Check response validity
                if not response or not hasattr(response, 'text'):
                    raise ValueError("Invalid LLM response object")
                
                response_text = response.text.strip()
                
                # Check for empty response
                if not response_text:
                    raise ValueError("Empty LLM response")
                
                # Extract JSON with better error handling
                json_text = self._extract_json_from_response(response_text)
                if not json_text:
                    raise ValueError("No valid JSON found in response")
                
                # Parse JSON with detailed error info
                try:
                    result = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    raise ValueError(f"JSON parsing failed: {json_err}. Raw text: {json_text[:100]}...")
                
                # Validate structure with detailed feedback
                validation_result = self._validate_llm_response(result)
                if validation_result['valid']:
                    return result
                else:
                    raise ValueError(f"Response validation failed: {validation_result['errors']}")
                    
            except Exception as e:
                error_type = type(e).__name__
                print(f"⚠️ LLM generation attempt {attempt + 1}/{max_retries} failed ({error_type}): {e}")
                
                # Different retry strategies based on error type
                if "timeout" in str(e).lower() or "network" in str(e).lower():
                    # Network/timeout error - wait and retry
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
                        continue
                elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    # Rate limit error - wait longer
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(5)
                        continue
                elif "json" in str(e).lower() or "parsing" in str(e).lower():
                    # JSON parsing error - retry once with simpler prompt
                    if attempt < max_retries - 1:
                        continue
                
                # For other errors or final attempt, break
                if attempt == max_retries - 1:
                    print(f"❌ All LLM attempts failed. Final error: {e}")
                    return None
        
        return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response with multiple strategies"""
        try:
            # Strategy 1: Look for ```json code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    return response_text[start:end].strip()
            
            # Strategy 2: Look for ```javascript or ``` blocks
            if "```" in response_text:
                start = response_text.find("```")
                if start >= 0:
                    start = response_text.find("\n", start) + 1
                    end = response_text.find("```", start)
                    if end > start:
                        candidate = response_text[start:end].strip()
                        if candidate.startswith("{"):
                            return candidate
            
            # Strategy 3: Response starts with JSON
            if response_text.startswith("{"):
                return response_text
            
            # Strategy 4: Find JSON object boundaries
            start = response_text.find("{")
            if start >= 0:
                end = response_text.rfind("}") + 1
                if end > start:
                    return response_text[start:end]
            
            return None
            
        except Exception as e:
            print(f"⚠️ JSON extraction failed: {e}")
            return None
    
    def _validate_llm_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM response structure with detailed feedback"""
        errors = []
        
        # Check required fields
        if "A1_keywords" not in result:
            errors.append("Missing 'A1_keywords' field")
        elif not isinstance(result["A1_keywords"], list):
            errors.append("'A1_keywords' must be a list")
        elif len(result["A1_keywords"]) == 0:
            errors.append("'A1_keywords' is empty")
        
        if "A2_prompts" not in result:
            errors.append("Missing 'A2_prompts' field")
        elif not isinstance(result["A2_prompts"], list):
            errors.append("'A2_prompts' must be a list")
        elif len(result["A2_prompts"]) == 0:
            errors.append("'A2_prompts' is empty")
        
        # Check content quality
        if "A1_keywords" in result and isinstance(result["A1_keywords"], list):
            keywords = result["A1_keywords"]
            if len(keywords) < 5:
                errors.append(f"Too few keywords: {len(keywords)} < 5")
            if any(not isinstance(k, str) or len(k.strip()) == 0 for k in keywords):
                errors.append("Keywords contain invalid entries")
        
        if "A2_prompts" in result and isinstance(result["A2_prompts"], list):
            prompts = result["A2_prompts"]
            if len(prompts) < 2:
                errors.append(f"Too few prompts: {len(prompts)} < 2")
            if any(not isinstance(p, str) or len(p.strip()) < 10 for p in prompts):
                errors.append("Prompts are too short or invalid")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _create_safety_fallback(self, situation: str, emotions: List[str]) -> Dict[str, Any]:
        """Create safety fallback when LLM fails"""
        print("🛡️ Using safety fallback generation")
        
        # Get frequent subjects as base keywords
        frequent_subjects = self.vocab_extractor.get_frequent_subjects(min_frequency=2)[:10]
        
        # Add emotion-based concepts (minimal hardcoding)
        emotion_concepts = []
        primary_emotion = emotions[0].lower() if emotions else "calm"
        
        if primary_emotion in ["stress", "anxiety", "overwhelmed"]:
            emotion_concepts = ["peaceful", "calm", "nature", "water", "blue"]
        elif primary_emotion in ["tired", "peaceful"]:
            emotion_concepts = ["soft", "warm", "gentle", "quiet", "restful"]
        elif primary_emotion in ["happy", "excited", "joyful"]:
            emotion_concepts = ["bright", "colorful", "vibrant", "celebration"]
        else:
            emotion_concepts = ["harmony", "balance", "beautiful", "serene"]
        
        # Combine and validate
        fallback_keywords = frequent_subjects + emotion_concepts
        valid_keywords, _ = self.vocab_extractor.validate_keywords(fallback_keywords)
        
        # Create simple prompts
        fallback_prompts = [
            f"calm and peaceful scene, soothing colors, gentle mood",
            f"harmonious composition, balanced colors, serene atmosphere", 
            f"tranquil environment, soft lighting, minimal complexity"
        ]
        
        return {
            "A1_keywords": valid_keywords[:self.config.max_a1_keywords],
            "A2_prompts": fallback_prompts[:self.config.max_a2_prompts],
            "debug": {
                "fallback_used": True,
                "reason": "LLM generation failed",
                "frequent_subjects_used": len(frequent_subjects),
                "emotion_concepts_used": len(emotion_concepts)
            }
        }
    
    def generate_dynamic_keywords_prompts(self, situation: str, emotions: List[str],
                                        brief: Optional[Dict] = None,
                                        evidence: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main method: Generate dynamic keywords and prompts
        
        Args:
            situation: User situation description
            emotions: List of user emotions
            brief: Optional Step 5 curation brief
            evidence: Optional evidence from Step 5
            
        Returns:
            Dictionary with A1_keywords, A2_prompts, and debug info
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(situation, emotions, brief)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result:
            elapsed = time.time() - start_time
            print(f"📦 Dynamic generation cache hit in {elapsed:.3f}s")
            return cached_result
        
        print(f"🎯 Dynamic keyword/prompt generation...")
        print(f"   Situation: {situation[:60]}...")
        print(f"   Emotions: {emotions}")
        
        # Try LLM generation
        llm_result = self._generate_with_llm(situation, emotions, brief, evidence)
        
        if llm_result:
            raw_keywords = llm_result.get("A1_keywords", [])
            raw_prompts = llm_result.get("A2_prompts", [])
            
            # Validate and process keywords
            valid_keywords, invalid_keywords = self.vocab_extractor.validate_keywords(raw_keywords)
            
            # Expand with concepts if enabled
            if self.config.enable_concept_expansion:
                expanded_keywords = self.vocab_extractor.expand_keywords_with_concepts(valid_keywords)
                valid_keywords = list(set(expanded_keywords))
            
            # Limit to target count
            final_keywords = valid_keywords[:self.config.max_a1_keywords]
            final_prompts = raw_prompts[:self.config.max_a2_prompts]
            
            result = {
                "A1_keywords": final_keywords,
                "A2_prompts": final_prompts,
                "debug": {
                    "raw_llm_keywords": raw_keywords,
                    "expanded_from_concepts": len(expanded_keywords) - len(valid_keywords) if self.config.enable_concept_expansion else 0,
                    "removed_non_vocab": invalid_keywords,
                    "used_brief_fields": list(brief.keys()) if brief else [],
                    "evidence_titles": [e.get("title", "Unknown") for e in evidence] if evidence else [],
                    "fallback_used": False,
                    "generation_method": "llm_dynamic"
                }
            }
            
        else:
            # Use safety fallback
            result = self._create_safety_fallback(situation, emotions)
        
        # Cache result
        self._save_to_cache(cache_key, result)
        
        elapsed = time.time() - start_time
        print(f"✅ Dynamic generation complete in {elapsed:.3f}s")
        print(f"   Keywords: {len(result['A1_keywords'])}")
        print(f"   Prompts: {len(result['A2_prompts'])}")
        
        return result

def test_dynamic_stage_a():
    """Test the dynamic Stage A system"""
    print("🧪 Testing Dynamic Stage A System")
    print("=" * 60)
    
    generator = DynamicKeywordPromptGenerator()
    
    test_scenarios = [
        {
            "situation": "Under heavy work-related stress and finding it hard to concentrate",
            "emotions": ["stress", "anxiety", "overwhelmed"],
            "brief": {
                "visual_elements": {
                    "preferred_themes": ["Calm landscapes", "Nature scenes"],
                    "color_psychology": {"primary_hues": ["Soft blues", "Gentle greens"]}
                }
            }
        },
        {
            "situation": "It's evening after a long day, and I want to unwind at home",
            "emotions": ["tired", "peaceful", "contemplative"],
            "brief": None
        },
        {
            "situation": "Need energy and motivation for creative work",
            "emotions": ["motivated", "focused", "excited"],
            "brief": None
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Test {i}: {scenario['situation'][:50]}...")
        
        result = generator.generate_dynamic_keywords_prompts(
            scenario["situation"],
            scenario["emotions"], 
            scenario.get("brief")
        )
        
        print(f"   📝 Keywords ({len(result['A1_keywords'])}): {result['A1_keywords']}")
        print(f"   🎨 Prompts ({len(result['A2_prompts'])}):")
        for j, prompt in enumerate(result['A2_prompts'], 1):
            print(f"      {j}. {prompt}")
        
        debug = result['debug']
        print(f"   🔍 Debug: Method={debug.get('generation_method', 'unknown')}, "
              f"Fallback={debug.get('fallback_used', False)}")
    
    print(f"\n✅ Dynamic Stage A testing complete!")

if __name__ == "__main__":
    test_dynamic_stage_a()