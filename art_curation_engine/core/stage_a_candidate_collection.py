#!/usr/bin/env python3
"""
Stage A: Candidate Collection System

Two-phase candidate collection for artwork recommendation:
- A1: Metadata OR expansion (fast keyword-based filtering)
- A2: CLIP text→image search (visual semantic search)

Based on plan.md specifications for Stage A implementation.
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv

# Load environment for CLIP model
load_dotenv(override=True)

# Import dynamic system
from .dynamic_stage_a import DynamicKeywordPromptGenerator

class StageACollector:
    """Stage A Candidate Collection System"""
    
    def __init__(self, 
                 metadata_path: str = "data/aic_sample/metadata.jsonl",
                 clip_index_dir: str = "indices/clip_faiss",
                 use_dynamic_system: bool = True):
        self.metadata_path = metadata_path
        self.clip_index_dir = Path(clip_index_dir)
        self.use_dynamic_system = use_dynamic_system
        
        # Metadata storage
        self.metadata_index = {}
        self.all_artwork_ids = []
        
        # CLIP components (loaded on demand)
        self.clip_model = None
        self.clip_preprocess = None
        self.faiss_index = None
        self.id_map = None
        self.artwork_index = None
        
        # Dynamic keyword/prompt generator
        self.dynamic_generator = None
        if self.use_dynamic_system:
            try:
                self.dynamic_generator = DynamicKeywordPromptGenerator()
                print("✅ Dynamic keyword/prompt generator initialized")
            except Exception as e:
                print(f"⚠️ Dynamic system initialization failed, falling back to hardcoded: {e}")
                self.use_dynamic_system = False
        
        self._load_metadata()
        print(f"✅ Stage A initialized with {len(self.all_artwork_ids)} artworks")
    
    def _load_metadata(self):
        """Load artwork metadata from JSONL file"""
        if not Path(self.metadata_path).exists():
            print(f"⚠️ Warning: Metadata file not found: {self.metadata_path}")
            return
            
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    artwork = json.loads(line)
                    artwork_id = artwork["id"]
                    self.metadata_index[artwork_id] = artwork
                    self.all_artwork_ids.append(artwork_id)
        
        print(f"📚 Loaded {len(self.all_artwork_ids)} artworks from metadata")
    
    def _load_clip_components(self):
        """Load CLIP model and FAISS index on demand"""
        if self.clip_model is not None:
            return  # Already loaded
            
        print("🔧 Loading CLIP model and FAISS index...")
        
        # Load CLIP model (using same logic as build_clip_index.py)
        try:
            import open_clip
            
            device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            
            # Set cache directory to avoid permission issues
            cache_dir = Path.home() / ".cache" / "huggingface"
            
            MODEL_NAME = "ViT-B-32"
            PRETRAINED = "laion2b_s34b_b79k"
            
            try:
                # Try to use local files first (same as build_clip_index.py)
                print("= Attempting to load model from local cache...")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME, 
                    pretrained=PRETRAINED, 
                    device=device,
                    cache_dir=str(cache_dir),
                )
                self.clip_model.eval()
                self.device = device
                print("✅ Model loaded successfully from local cache")
                
            except Exception as e:
                print(f"Local model loading failed: {e}")
                print("= Trying without pretrained weights...")
                
                # Fallback: try without pretrained weights
                try:
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                        MODEL_NAME, 
                        pretrained=None,  # No pretrained weights
                        device=device
                    )
                    self.clip_model.eval()
                    self.device = device
                    print("⚠️ Model loaded without pretrained weights (random initialization)")
                except Exception as e2:
                    # Last resort: try a different model that's definitely local
                    print("= Trying OpenAI CLIP model...")
                    try:
                        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                            "ViT-B-32", 
                            pretrained="openai",  # This should be more stable
                            device=device
                        )
                        self.clip_model.eval()
                        self.device = device
                        print("✅ Loaded OpenAI CLIP model instead")
                    except Exception as e3:
                        raise RuntimeError(f"All model loading attempts failed: {e3}")
            
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            return
        
        # Load FAISS index and mappings
        try:
            import faiss
            
            faiss_path = self.clip_index_dir / "faiss.index"
            id_map_path = self.clip_index_dir / "id_map.json"
            artwork_index_path = self.clip_index_dir / "artwork_index.json"
            
            if not all(p.exists() for p in [faiss_path, id_map_path, artwork_index_path]):
                print("⚠️ CLIP index files not found. Please run build_clip_index.py first.")
                return
                
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            with open(id_map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
                # Convert string keys to int for FAISS row indexing
                self.id_map = {int(k): v for k, v in self.id_map.items()}
            
            with open(artwork_index_path, "r", encoding="utf-8") as f:
                self.artwork_index = json.load(f)
                # Convert keys to int for artwork_id lookup
                self.artwork_index = {int(k): v for k, v in self.artwork_index.items()}
                
            print(f"✅ FAISS index loaded with {self.faiss_index.ntotal} embeddings")
            
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    
    def generate_open_keywords(self, situation: str, emotions: List[str]) -> List[str]:
        """
        Generate open OR keywords from user context
        
        A1 phase: Convert situation + emotions into broad keyword set
        for metadata filtering
        """
        # Try dynamic system first
        if self.use_dynamic_system and self.dynamic_generator:
            try:
                result = self.dynamic_generator.generate_dynamic_keywords_prompts(
                    situation=situation,
                    emotions=emotions
                )
                dynamic_keywords = result.get("A1_keywords", [])
                
                if dynamic_keywords:
                    print(f"🔑 Generated {len(dynamic_keywords)} open keywords: {dynamic_keywords}")
                    return dynamic_keywords
                else:
                    print("⚠️ Dynamic system returned empty keywords, using fallback")
                    
            except Exception as e:
                print(f"⚠️ Dynamic keyword generation failed: {e}, using fallback")
        
        # Fallback to hardcoded system
        return self._generate_hardcoded_keywords(situation, emotions)
    
    def _generate_hardcoded_keywords(self, situation: str, emotions: List[str]) -> List[str]:
        """
        Fallback hardcoded keyword generation (original implementation)
        """
        keywords = set()
        
        # Emotion-based keywords
        emotion_mapping = {
            "stress": ["calm", "peaceful", "serene", "quiet", "relaxing"],
            "anxiety": ["soothing", "gentle", "tranquil", "soft", "harmonious"],
            "overwhelmed": ["simple", "minimal", "clear", "spacious", "open"],
            "tired": ["restful", "comfortable", "warm", "cozy", "still"],
            "peaceful": ["nature", "landscape", "water", "sky", "garden"],
            "contemplative": ["abstract", "meditative", "spiritual", "reflective"],
            "sad": ["uplifting", "hopeful", "bright", "colorful", "joyful"],
            "happy": ["vibrant", "energetic", "dynamic", "celebration", "light"],
            "excited": ["bold", "dramatic", "expressive", "movement", "passion"],
            "lonely": ["connection", "human", "community", "together", "warmth"],
            "angry": ["cooling", "calming", "neutral", "balanced", "harmony"],
            "confused": ["clarity", "structure", "organized", "defined", "clear"]
        }
        
        for emotion in emotions:
            emotion_lower = emotion.lower()
            if emotion_lower in emotion_mapping:
                keywords.update(emotion_mapping[emotion_lower])
        
        # Situation-based keywords
        situation_lower = situation.lower()
        
        # Work/stress related
        if any(word in situation_lower for word in ["work", "stress", "concentrate", "focus"]):
            keywords.update(["nature", "landscape", "blue", "green", "water", "forest", "calm"])
        
        # Evening/relaxation related  
        if any(word in situation_lower for word in ["evening", "unwind", "relax", "home", "rest"]):
            keywords.update(["warm", "golden", "sunset", "interior", "cozy", "soft", "gentle"])
        
        # Morning/energy related
        if any(word in situation_lower for word in ["morning", "energy", "start", "begin"]):
            keywords.update(["bright", "fresh", "sunrise", "flowers", "spring", "vibrant"])
        
        # Social/relationship related
        if any(word in situation_lower for word in ["friend", "family", "social", "together"]):
            keywords.update(["people", "gathering", "celebration", "community", "portrait"])
        
        # Add general calming/therapeutic keywords
        keywords.update(["art", "beautiful", "peaceful", "harmony", "balance"])
        
        # Convert to list and limit size
        final_keywords = list(keywords)[:20]  # Limit to top 20 keywords
        
        print(f"🔑 Generated {len(final_keywords)} hardcoded keywords: {final_keywords}")
        return final_keywords
    
    def score_artwork_a1(self, artwork_id: int, keywords: List[str]) -> float:
        """
        Score artwork for A1 phase based on metadata matching
        
        Scoring:
        - subject_titles intersection ≥ 1: base score
        - style_title context match: +0.2
        - description/alt_text with calming words: +0.2
        """
        artwork = self.metadata_index.get(artwork_id)
        if not artwork:
            return 0.0
        
        score = 0.0
        
        # Check subject_titles intersection
        subject_titles = artwork.get("subject_titles", [])
        if subject_titles:
            subject_text = " ".join(str(s).lower() for s in subject_titles)
            
            matches = 0
            for keyword in keywords:
                if keyword.lower() in subject_text:
                    matches += 1
            
            if matches > 0:
                score = 0.5 + min(0.3, matches * 0.1)  # Base score + bonus for multiple matches
        
        # Style context bonus
        style_title = artwork.get("style_title", "")
        if style_title:
            style_lower = style_title.lower()
            calming_styles = ["impressionism", "minimalism", "abstract", "landscape", "pastoral"]
            if any(style in style_lower for style in calming_styles):
                score += 0.2
        
        # Description/alt_text bonus
        description = artwork.get("short_description", "") or ""
        alt_text = artwork.get("thumbnail", {}).get("alt_text", "") or ""
        combined_text = (description + " " + alt_text).lower()
        
        calming_words = ["calm", "soft", "dreamy", "peaceful", "gentle", "serene", "tranquil", "soothing"]
        if any(word in combined_text for word in calming_words):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def run_a1_metadata_expansion(self, situation: str, emotions: List[str], 
                                 top_k_cap: int = 200) -> List[Tuple[int, float]]:
        """
        A1: Metadata OR expansion phase
        
        Returns:
            List of (artwork_id, fast_score) tuples, sorted by score descending
        """
        print(f"🔍 A1: Metadata OR expansion (target: {top_k_cap} candidates)")
        
        # Generate keywords
        keywords = self.generate_open_keywords(situation, emotions)
        
        # Score all artworks
        scored_candidates = []
        for artwork_id in self.all_artwork_ids:
            score = self.score_artwork_a1(artwork_id, keywords)
            if score > 0:  # Only include artworks with some match
                scored_candidates.append((artwork_id, score))
        
        # Sort by score descending and apply cap
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        final_candidates = scored_candidates[:top_k_cap]
        
        print(f"✅ A1 complete: {len(final_candidates)} candidates (from {len(scored_candidates)} matches)")
        return final_candidates
    
    def generate_clip_prompts(self, situation: str, emotions: List[str]) -> List[str]:
        """
        Generate 3-5 CLIP search prompts based on user context
        
        A2 phase: Create visual search prompts that capture the emotional
        and situational needs through visual descriptions
        """
        # Try dynamic system first
        if self.use_dynamic_system and self.dynamic_generator:
            try:
                result = self.dynamic_generator.generate_dynamic_keywords_prompts(
                    situation=situation,
                    emotions=emotions
                )
                dynamic_prompts = result.get("A2_prompts", [])
                
                if dynamic_prompts:
                    print(f"🎨 Generated {len(dynamic_prompts)} CLIP prompts:")
                    for i, prompt in enumerate(dynamic_prompts, 1):
                        print(f"   {i}. {prompt[:60]}...")
                    return dynamic_prompts
                else:
                    print("⚠️ Dynamic system returned empty prompts, using fallback")
                    
            except Exception as e:
                print(f"⚠️ Dynamic prompt generation failed: {e}, using fallback")
        
        # Fallback to hardcoded system
        return self._generate_hardcoded_prompts(situation, emotions)
    
    def _generate_hardcoded_prompts(self, situation: str, emotions: List[str]) -> List[str]:
        """
        Fallback hardcoded prompt generation (original implementation)
        """
        prompts = []
        
        # Base emotional tone
        if any(emotion in emotions for emotion in ["stress", "anxiety", "overwhelmed"]):
            prompts.append("soft calming blue and green nature scene, peaceful landscape, gentle lighting")
            prompts.append("tranquil water reflection, minimal visual clutter, soothing colors")
        
        if any(emotion in emotions for emotion in ["tired", "peaceful", "contemplative"]):
            prompts.append("quiet serene environment, muted tones, restful composition")
            prompts.append("meditative abstract patterns, low saturation, harmonious balance")
        
        if any(emotion in emotions for emotion in ["sad", "lonely"]):
            prompts.append("warm uplifting colors, hopeful imagery, human connection")
            prompts.append("bright cheerful scene, community gathering, positive energy")
        
        if any(emotion in emotions for emotion in ["happy", "excited", "joyful"]):
            prompts.append("vibrant colorful celebration, dynamic movement, energetic composition")
            prompts.append("bright sunny day, flowers and nature, joyful atmosphere")
        
        # Situation-specific prompts
        situation_lower = situation.lower()
        
        if "evening" in situation_lower or "unwind" in situation_lower:
            prompts.append("warm golden evening light, cozy interior, soft textures")
        
        if "work" in situation_lower or "concentrate" in situation_lower:
            prompts.append("organized clean space, focused lighting, minimal distractions")
        
        if "morning" in situation_lower:
            prompts.append("fresh morning sunlight, new beginnings, bright clean energy")
        
        # Ensure we have exactly 4 prompts (recommended by plan.md)
        if len(prompts) < 4:
            # Add generic calming prompts
            prompts.extend([
                "peaceful natural environment, soft organic shapes, calming atmosphere",
                "harmonious color palette, balanced composition, therapeutic visual elements"
            ])
        
        # Take first 4 prompts
        final_prompts = prompts[:4]
        
        print(f"🎨 Generated {len(final_prompts)} CLIP prompts:")
        for i, prompt in enumerate(final_prompts, 1):
            print(f"   {i}. {prompt}")
        
        return final_prompts
    
    @torch.no_grad()
    def search_clip_prompt(self, prompt: str, top_k: int = 140) -> List[Tuple[int, float]]:
        """
        Search FAISS index with a single CLIP text prompt
        
        Returns:
            List of (artwork_id, similarity_score) tuples
        """
        if self.clip_model is None or self.faiss_index is None:
            return []
        
        # Encode text prompt
        import open_clip
        text_tokens = open_clip.tokenize([prompt]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Search FAISS index
        query_vector = text_features.detach().cpu().numpy().astype("float32")
        similarities, indices = self.faiss_index.search(query_vector, top_k)
        
        # Convert FAISS results to artwork IDs
        results = []
        for i, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0])):
            if faiss_idx >= 0 and faiss_idx in self.id_map:
                artwork_id = self.id_map[faiss_idx]["artwork_id"]
                results.append((artwork_id, float(similarity)))
        
        return results
    
    def run_a2_clip_search(self, situation: str, emotions: List[str],
                          per_query_topk: int = 140, cap_after_union: int = 150) -> Tuple[List[int], Dict[int, float]]:
        """
        A2: CLIP text→image search phase
        
        Returns:
            Tuple of (artwork_ids, clip_scores)
        """
        print(f"🎯 A2: CLIP text→image search (per_query: {per_query_topk}, cap: {cap_after_union})")
        
        # Ensure CLIP components are loaded
        self._load_clip_components()
        if self.clip_model is None:
            print("❌ CLIP model not available, skipping A2")
            return [], {}
        
        # Generate prompts
        prompts = self.generate_clip_prompts(situation, emotions)
        
        # Search with each prompt
        all_results = {}  # artwork_id -> max_similarity
        
        for i, prompt in enumerate(prompts, 1):
            print(f"   🔍 Searching prompt {i}/{len(prompts)}: {prompt[:60]}...")
            
            results = self.search_clip_prompt(prompt, per_query_topk)
            
            for artwork_id, similarity in results:
                if artwork_id in all_results:
                    # Keep maximum similarity across prompts
                    all_results[artwork_id] = max(all_results[artwork_id], similarity)
                else:
                    all_results[artwork_id] = similarity
        
        # Sort by similarity and apply cap
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        final_results = sorted_results[:cap_after_union]
        
        artwork_ids = [artwork_id for artwork_id, _ in final_results]
        clip_scores = {artwork_id: score for artwork_id, score in final_results}
        
        print(f"✅ A2 complete: {len(artwork_ids)} candidates from {len(all_results)} total matches")
        return artwork_ids, clip_scores
    
    def merge_a1_a2_results(self, a1_candidates: List[Tuple[int, float]], 
                           a2_ids: List[int], a2_scores: Dict[int, float],
                           final_target: int = 150) -> Tuple[List[int], Dict[int, float]]:
        """
        Merge A1 metadata and A2 CLIP results
        
        Strategy from plan.md:
        - A2 (visual semantic) leads
        - A1 (metadata match) provides light correction
        - A2 hits get base score 1.0 + A1 bonus
        - A1-only hits get small diversity bonus
        """
        print(f"🔗 Merging A1 and A2 results (target: {final_target})")
        
        merged_scores = {}
        a1_dict = {aid: score for aid, score in a1_candidates}
        
        # Process A2 candidates (primary)
        for artwork_id in a2_ids:
            # Base score from A2 CLIP similarity 
            clip_score = a2_scores.get(artwork_id, 0.0)
            merged_score = 1.0  # Base score for A2 hits
            
            # Add A1 metadata bonus if available
            if artwork_id in a1_dict:
                a1_bonus = min(0.1 * a1_dict[artwork_id], 0.5)
                merged_score += a1_bonus
            
            merged_scores[artwork_id] = merged_score
        
        # Add diversity from A1-only candidates (small boost)
        a1_only = [aid for aid, _ in a1_candidates if aid not in merged_scores]
        for artwork_id in a1_only[:min(10, final_target // 10)]:  # Add up to 10% diversity
            merged_scores[artwork_id] = 0.05  # Small diversity score
        
        # Sort and apply final cap
        sorted_merged = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        final_candidates = sorted_merged[:final_target]
        
        final_ids = [artwork_id for artwork_id, _ in final_candidates]
        final_scores = {artwork_id: score for artwork_id, score in final_candidates}
        
        print(f"✅ Merge complete: {len(final_ids)} final candidates")
        return final_ids, final_scores
    
    def collect_candidates(self, situation: str, emotions: List[str],
                          mode: str = "balanced") -> Dict[str, Any]:
        """
        Run complete Stage A candidate collection pipeline
        
        Args:
            situation: User situation description
            emotions: List of user emotions  
            mode: "budget" (120), "balanced" (150), or "quality-max" (180)
            
        Returns:
            Complete Stage A results with candidates, scores, and debug info
        """
        print(f"\n🎭 Stage A: Candidate Collection ({mode} mode)")
        print(f"📝 Situation: {situation}")
        print(f"😊 Emotions: {emotions}")
        print("=" * 80)
        
        # Set parameters based on mode
        mode_params = {
            "budget": {"target": 120, "a1_cap": 180, "a2_topk": 125, "a2_cap": 120},
            "balanced": {"target": 150, "a1_cap": 200, "a2_topk": 140, "a2_cap": 150},
            "quality-max": {"target": 180, "a1_cap": 220, "a2_topk": 155, "a2_cap": 180}
        }
        
        params = mode_params.get(mode, mode_params["balanced"])
        
        # A1: Metadata OR expansion
        a1_candidates = self.run_a1_metadata_expansion(
            situation, emotions, top_k_cap=params["a1_cap"]
        )
        
        # A2: CLIP text→image search
        a2_ids, a2_scores = self.run_a2_clip_search(
            situation, emotions, 
            per_query_topk=params["a2_topk"], 
            cap_after_union=params["a2_cap"]
        )
        
        # Merge results
        final_ids, merged_scores = self.merge_a1_a2_results(
            a1_candidates, a2_ids, a2_scores, final_target=params["target"]
        )
        
        # Prepare debug information
        keywords = self.generate_open_keywords(situation, emotions)
        prompts = self.generate_clip_prompts(situation, emotions)
        
        # Package results
        results = {
            "final_stageA_ids": final_ids,
            "clip_scores": {str(aid): a2_scores.get(aid, 0.0) for aid in final_ids},
            "debug": {
                "open_keywords": keywords,
                "prompts": prompts,
                "A1_hits": len(a1_candidates),
                "A2_union_hits": len(a2_ids),
                "mode": mode,
                "parameters": params
            }
        }
        
        print(f"\n📊 Stage A Results ({mode}):")
        print(f"   Final candidates: {len(final_ids)}")
        print(f"   A1 metadata hits: {len(a1_candidates)}")
        print(f"   A2 CLIP hits: {len(a2_ids)}")
        print(f"   Generated keywords: {len(keywords)}")
        print(f"   CLIP prompts: {len(prompts)}")
        
        return results

def test_stage_a():
    """Test Stage A candidate collection system"""
    collector = StageACollector()
    
    # Test with work stress scenario
    results = collector.collect_candidates(
        situation="Under heavy work-related stress and finding it hard to concentrate",
        emotions=["stress", "anxiety", "overwhelmed"],
        mode="balanced"
    )
    
    print("\n" + "=" * 80)
    print("🧪 Stage A Test Results")
    print("=" * 80)
    
    final_ids = results["final_stageA_ids"]
    print(f"Collected {len(final_ids)} candidates")
    print(f"Top 10 candidate IDs: {final_ids[:10]}")
    
    debug = results["debug"]
    print(f"\nKeywords: {debug['open_keywords'][:10]}")
    print(f"A1 hits: {debug['A1_hits']}")
    print(f"A2 hits: {debug['A2_union_hits']}")
    
    return results

if __name__ == "__main__":
    test_stage_a()