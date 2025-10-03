#!/usr/bin/env python3

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)

# LangChain imports
from langchain_fireworks import ChatFireworks
from langchain.schema import HumanMessage

# Import prompts
from .llm_prompts import Step6Prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    """Individual artwork scoring result"""
    artwork_id: str
    emotional_fit: float
    narrative_fit: float
    subject_fit: float
    palette_fit: float
    style_fit: float
    evidence_alignment: float
    justification: str = ""
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted average score"""
        weights = {
            'emotional_fit': 0.25,
            'narrative_fit': 0.20,
            'subject_fit': 0.15,
            'palette_fit': 0.20,
            'style_fit': 0.10,
            'evidence_alignment': 0.10
        }
        
        return (
            self.emotional_fit * weights['emotional_fit'] +
            self.narrative_fit * weights['narrative_fit'] +
            self.subject_fit * weights['subject_fit'] +
            self.palette_fit * weights['palette_fit'] +
            self.style_fit * weights['style_fit'] +
            self.evidence_alignment * weights['evidence_alignment']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'artwork_id': self.artwork_id,
            'rerank_score': self.weighted_score,
            'scoring_breakdown': {
                'emotional_fit': self.emotional_fit,
                'narrative_fit': self.narrative_fit,
                'subject_fit': self.subject_fit,
                'palette_fit': self.palette_fit,
                'style_fit': self.style_fit,
                'evidence_alignment': self.evidence_alignment
            },
            'justification': self.justification
        }

class Step6LLMReranker:
    """
    Step 6 LLM Reranking System
    
    Reranks 150 Stage A candidates to 30 final recommendations using:
    - Multi-dimensional LLM scoring (6 dimensions)
    - Batch processing for efficiency  
    - Caching for performance
    - MMR diversification for variety
    """
    
    def __init__(self, 
                 model_name: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
                 batch_size: int = 15,
                 max_workers: int = 3,
                 cache_ttl_hours: int = 24):
        """
        Initialize Step 6 LLM Reranker
        
        Args:
            model_name: LLM model for scoring
            batch_size: Number of artworks per batch
            max_workers: Max parallel batches
            cache_ttl_hours: Cache time-to-live in hours
        """
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize cache
        self.score_cache = {}
        self.justification_cache = {}
        self.cache_file = "storage/step6_score_cache.json" 
        self.justification_cache_file = "storage/step6_justification_cache.json"
        self._load_cache()
        self._load_justification_cache()
        
        logger.info(f"Step 6 LLM Reranker initialized with model: {model_name}")
    
    def _initialize_llm(self) -> ChatFireworks:
        """Initialize LLM with error handling"""
        try:
            api_key = os.getenv('FIREWORKS_API_KEY')
            if not api_key:
                raise ValueError("FIREWORKS_API_KEY not found in environment")
            
            llm = ChatFireworks(
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=4000,
                fireworks_api_key=api_key
            )
            
            logger.info("LLM initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _load_cache(self) -> None:
        """Load score cache from disk"""
        try:
            os.makedirs("storage", exist_ok=True)
            
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Filter expired entries
                current_time = datetime.now()
                self.score_cache = {
                    k: v for k, v in cache_data.items()
                    if datetime.fromisoformat(v['timestamp']) + self.cache_ttl > current_time
                }
                
                logger.info(f"Loaded {len(self.score_cache)} cached scores")
            else:
                self.score_cache = {}
                logger.info("No cache file found, starting with empty cache")
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.score_cache = {}
    
    def _save_cache(self) -> None:
        """Save score cache to disk"""
        try:
            os.makedirs("storage", exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.score_cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_justification_cache(self) -> None:
        """Load justification cache from disk"""
        try:
            if os.path.exists(self.justification_cache_file):
                with open(self.justification_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Filter expired entries
                current_time = datetime.now()
                self.justification_cache = {
                    k: v for k, v in cache_data.items()
                    if datetime.fromisoformat(v['timestamp']) + self.cache_ttl > current_time
                }
                
                logger.info(f"Loaded {len(self.justification_cache)} cached justifications")
            else:
                self.justification_cache = {}
                logger.info("No justification cache file found, starting with empty cache")
                
        except Exception as e:
            logger.warning(f"Failed to load justification cache: {e}")
            self.justification_cache = {}
    
    def _save_justification_cache(self) -> None:
        """Save justification cache to disk"""
        try:
            os.makedirs("storage", exist_ok=True)
            
            with open(self.justification_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.justification_cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save justification cache: {e}")
    
    def _get_cache_key(self, brief_hash: str, artwork_id: str) -> str:
        """Generate cache key for brief-artwork combination"""
        return f"{brief_hash}_{artwork_id}"
    
    def _hash_brief(self, rag_brief: Dict[str, Any]) -> str:
        """Generate hash for RAG brief for caching"""
        # Create stable hash from key brief elements
        brief_str = json.dumps({
            'situation_analysis': rag_brief.get('situation_analysis', ''),
            'curation_strategy': rag_brief.get('curation_strategy', ''),
            'visual_elements': rag_brief.get('visual_elements', {}),
            'curatorial_goals': rag_brief.get('curatorial_goals', [])
        }, sort_keys=True)
        
        return hashlib.md5(brief_str.encode()).hexdigest()[:16]
    
    def rerank_candidates(self, 
                         rag_brief: Dict[str, Any], 
                         candidates: List[Dict[str, Any]],
                         target_count: int = 30) -> Dict[str, Any]:
        """
        Main reranking function: 150 candidates → 30 final recommendations
        
        Args:
            rag_brief: Step 5 generated RAG brief
            candidates: 150 Stage A candidate artworks
            target_count: Number of final recommendations (default 30)
            
        Returns:
            Dictionary with final recommendations and metadata
        """
        
        start_time = datetime.now()
        logger.info(f"Starting Step 6 reranking: {len(candidates)} → {target_count} candidates")
        
        # Validate inputs
        if not rag_brief or not candidates:
            raise ValueError("RAG brief and candidates are required")
        
        if len(candidates) == 0:
            logger.warning("No candidates provided")
            return {'final_recommendations': [], 'metadata': {}}
        
        # Generate brief hash for caching
        brief_hash = self._hash_brief(rag_brief)
        
        # Phase 1: Score all candidates
        logger.info("Phase 1: Scoring candidates with LLM...")
        all_scores = self._score_all_candidates(rag_brief, candidates, brief_hash)
        
        # Phase 2: Select top candidates with diversity
        logger.info("Phase 2: Selecting diverse top candidates...")
        final_recommendations = self._select_diverse_top_candidates(
            all_scores, candidates, target_count
        )
        
        # Phase 3: Generate justifications for final selections
        logger.info("Phase 3: Generating justifications...")
        self._add_justifications(final_recommendations, rag_brief)
        
        # Save caches
        self._save_cache()
        self._save_justification_cache()
        
        # Prepare result
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            'final_recommendations': [rec.to_dict() for rec in final_recommendations],
            'metadata': {
                'input_count': len(candidates),
                'output_count': len(final_recommendations),
                'processing_time_seconds': processing_time,
                'brief_hash': brief_hash,
                'timestamp': end_time.isoformat(),
                'model_used': self.model_name,
                'cache_hits': sum(1 for score in all_scores if hasattr(score, '_from_cache'))
            }
        }
        
        logger.info(f"Step 6 completed in {processing_time:.2f}s - {len(final_recommendations)} recommendations")
        return result
    
    def _score_all_candidates(self, 
                            rag_brief: Dict[str, Any], 
                            candidates: List[Dict[str, Any]], 
                            brief_hash: str) -> List[ScoringResult]:
        """Score all candidates using batch processing and caching"""
        
        # Check cache first
        cached_scores = []
        uncached_candidates = []
        
        for candidate in candidates:
            artwork_id = candidate.get('artwork_id', candidate.get('id', ''))
            cache_key = self._get_cache_key(brief_hash, artwork_id)
            
            if cache_key in self.score_cache:
                # Load from cache
                cached_data = self.score_cache[cache_key]
                score = ScoringResult(
                    artwork_id=artwork_id,
                    emotional_fit=cached_data['emotional_fit'],
                    narrative_fit=cached_data['narrative_fit'],
                    subject_fit=cached_data['subject_fit'],
                    palette_fit=cached_data['palette_fit'],
                    style_fit=cached_data['style_fit'],
                    evidence_alignment=cached_data['evidence_alignment']
                )
                score._from_cache = True
                cached_scores.append(score)
            else:
                uncached_candidates.append(candidate)
        
        logger.info(f"Cache hits: {len(cached_scores)}, Need scoring: {len(uncached_candidates)}")
        
        # Score uncached candidates in batches
        new_scores = []
        if uncached_candidates:
            new_scores = self._score_candidates_batch(rag_brief, uncached_candidates, brief_hash)
        
        # Combine all scores
        all_scores = cached_scores + new_scores
        
        # Sort by artwork_id to maintain consistent ordering
        all_scores.sort(key=lambda x: x.artwork_id)
        
        return all_scores
    
    def _score_candidates_batch(self, 
                              rag_brief: Dict[str, Any], 
                              candidates: List[Dict[str, Any]], 
                              brief_hash: str) -> List[ScoringResult]:
        """Score candidates using batch processing"""
        
        # Create batches
        batches = [
            candidates[i:i + self.batch_size] 
            for i in range(0, len(candidates), self.batch_size)
        ]
        
        logger.info(f"Processing {len(batches)} batches with {self.max_workers} workers")
        
        all_scores = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch scoring tasks
            future_to_batch = {
                executor.submit(self._score_single_batch, rag_brief, batch, brief_hash): batch
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_scores = future.result()
                    all_scores.extend(batch_scores)
                    logger.info(f"Completed batch of {len(batch)} candidates")
                except Exception as e:
                    logger.error(f"Batch scoring failed: {e}")
                    # Fallback: create default scores
                    fallback_scores = self._create_fallback_scores(batch)
                    all_scores.extend(fallback_scores)
        
        return all_scores
    
    def _score_single_batch(self, 
                          rag_brief: Dict[str, Any], 
                          candidate_batch: List[Dict[str, Any]], 
                          brief_hash: str) -> List[ScoringResult]:
        """Score a single batch of candidates using LLM"""
        
        try:
            # Generate scoring prompt
            prompt = Step6Prompts.get_batch_scoring_prompt(rag_brief, candidate_batch)
            
            # Call LLM
            logger.info(f"Scoring batch of {len(candidate_batch)} artworks with LLM...")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response
            scores = self._parse_llm_scores(response.content, candidate_batch, brief_hash)
            
            logger.info(f"Successfully scored {len(scores)} artworks")
            return scores
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            # Fallback to mock scoring
            return self._create_fallback_scores(candidate_batch)
    
    def _parse_llm_scores(self, 
                         llm_response: str, 
                         candidates: List[Dict[str, Any]], 
                         brief_hash: str) -> List[ScoringResult]:
        """Parse LLM response into ScoringResult objects"""
        
        try:
            # Extract JSON from response (handle potential markdown formatting)
            response_text = llm_response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text
            
            # Parse JSON
            parsed_response = json.loads(json_text)
            scores_data = parsed_response.get('scores', [])
            
            if len(scores_data) != len(candidates):
                logger.warning(f"Score count mismatch: got {len(scores_data)}, expected {len(candidates)}")
            
            scores = []
            for i, score_data in enumerate(scores_data):
                if i >= len(candidates):
                    break
                    
                candidate = candidates[i]
                artwork_id = candidate.get('artwork_id', candidate.get('id', f'unknown_{i}'))
                
                # Extract scores with validation
                score = ScoringResult(
                    artwork_id=artwork_id,
                    emotional_fit=self._validate_score(score_data.get('emotional_fit', 0.5)),
                    narrative_fit=self._validate_score(score_data.get('narrative_fit', 0.5)),
                    subject_fit=self._validate_score(score_data.get('subject_fit', 0.5)),
                    palette_fit=self._validate_score(score_data.get('palette_fit', 0.5)),
                    style_fit=self._validate_score(score_data.get('style_fit', 0.5)),
                    evidence_alignment=self._validate_score(score_data.get('evidence_alignment', 0.5))
                )
                
                scores.append(score)
                
                # Cache the score
                cache_key = self._get_cache_key(brief_hash, artwork_id)
                self.score_cache[cache_key] = {
                    'emotional_fit': score.emotional_fit,
                    'narrative_fit': score.narrative_fit,
                    'subject_fit': score.subject_fit,
                    'palette_fit': score.palette_fit,
                    'style_fit': score.style_fit,
                    'evidence_alignment': score.evidence_alignment,
                    'timestamp': datetime.now().isoformat()
                }
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to parse LLM scores: {e}")
            logger.error(f"LLM response: {llm_response[:500]}...")
            
            # Fallback to default scores
            return self._create_fallback_scores(candidates)
    
    def _validate_score(self, score: Any) -> float:
        """Validate and clamp score to 0.0-1.0 range"""
        try:
            float_score = float(score)
            return max(0.0, min(1.0, float_score))
        except (ValueError, TypeError):
            logger.warning(f"Invalid score value: {score}, using 0.5")
            return 0.5
    
    def _create_fallback_scores(self, candidates: List[Dict[str, Any]]) -> List[ScoringResult]:
        """Create fallback scores when LLM scoring fails"""
        
        fallback_scores = []
        for candidate in candidates:
            artwork_id = candidate.get('artwork_id', candidate.get('id', ''))
            
            # Use Stage A score as base, add some variation
            stage_a_score = candidate.get('score', 0.5)
            base_score = min(0.9, max(0.1, stage_a_score))
            
            # Add small random variation to each dimension
            variation = np.random.uniform(-0.1, 0.1, 6)
            scores_array = np.clip([base_score] * 6 + variation, 0.0, 1.0)
            
            fallback_score = ScoringResult(
                artwork_id=artwork_id,
                emotional_fit=scores_array[0],
                narrative_fit=scores_array[1],
                subject_fit=scores_array[2],
                palette_fit=scores_array[3],
                style_fit=scores_array[4],
                evidence_alignment=scores_array[5]
            )
            
            fallback_scores.append(fallback_score)
        
        logger.warning(f"Created {len(fallback_scores)} fallback scores")
        return fallback_scores
    
    def _select_diverse_top_candidates(self, 
                                     scores: List[ScoringResult], 
                                     candidates: List[Dict[str, Any]], 
                                     target_count: int) -> List[ScoringResult]:
        """Select top candidates with diversity using MMR approach"""
        
        # Create candidate lookup by artwork_id
        candidate_lookup = {c.get('artwork_id', ''): c for c in candidates}
        
        # Sort by weighted score and take top 60% as candidate pool
        sorted_scores = sorted(scores, key=lambda x: x.weighted_score, reverse=True)
        candidate_pool_size = max(target_count, min(len(sorted_scores), int(len(sorted_scores) * 0.6)))
        candidate_pool = sorted_scores[:candidate_pool_size]
        
        logger.info(f"MMR selection from pool of {len(candidate_pool)} high-scoring candidates")
        
        # Apply MMR for diversity
        selected = self._apply_mmr_selection(candidate_pool, candidate_lookup, target_count)
        
        logger.info(f"Selected {len(selected)} diverse candidates (score range: {selected[0].weighted_score:.3f} - {selected[-1].weighted_score:.3f})")
        
        return selected
    
    def _apply_mmr_selection(self, 
                           candidate_pool: List[ScoringResult],
                           candidate_lookup: Dict[str, Dict[str, Any]],
                           target_count: int,
                           lambda_param: float = 0.7) -> List[ScoringResult]:
        """
        Apply Maximal Marginal Relevance (MMR) for diverse selection
        
        Args:
            candidate_pool: High-scoring candidates to select from
            candidate_lookup: Metadata for candidates
            target_count: Number of candidates to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        
        if len(candidate_pool) <= target_count:
            return candidate_pool
        
        selected = []
        remaining = candidate_pool.copy()
        
        # Always start with the highest scoring candidate
        selected.append(remaining.pop(0))
        
        # Iteratively select candidates that maximize MMR score
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_mmr_score = -1
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Calculate relevance score (normalized)
                relevance = candidate.weighted_score
                
                # Calculate diversity score (1 - max_similarity with selected)
                diversity = 1.0 - self._calculate_max_similarity(
                    candidate, selected, candidate_lookup
                )
                
                # Calculate MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _calculate_max_similarity(self, 
                                candidate: ScoringResult,
                                selected: List[ScoringResult],
                                candidate_lookup: Dict[str, Dict[str, Any]]) -> float:
        """Calculate maximum similarity between candidate and selected artworks"""
        
        candidate_metadata = candidate_lookup.get(candidate.artwork_id, {})
        max_similarity = 0.0
        
        for selected_artwork in selected:
            selected_metadata = candidate_lookup.get(selected_artwork.artwork_id, {})
            similarity = self._calculate_similarity(candidate_metadata, selected_metadata)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_similarity(self, 
                            artwork1: Dict[str, Any], 
                            artwork2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two artworks based on multiple features
        
        Returns similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        
        similarity_scores = []
        
        # Style similarity (0.3 weight)
        style1 = set(artwork1.get('style_titles', []))
        style2 = set(artwork2.get('style_titles', []))
        if style1 or style2:
            style_similarity = len(style1.intersection(style2)) / max(len(style1.union(style2)), 1)
            similarity_scores.append(('style', style_similarity, 0.3))
        
        # Subject similarity (0.2 weight)
        subject1 = set(str(s) for s in artwork1.get('subject_ids', []))
        subject2 = set(str(s) for s in artwork2.get('subject_ids', []))
        if subject1 or subject2:
            subject_similarity = len(subject1.intersection(subject2)) / max(len(subject1.union(subject2)), 1)
            similarity_scores.append(('subject', subject_similarity, 0.2))
        
        # Medium similarity (0.2 weight)
        medium1 = artwork1.get('medium_display', '').lower()
        medium2 = artwork2.get('medium_display', '').lower()
        if medium1 and medium2:
            # Simple string similarity for medium
            medium_similarity = 1.0 if medium1 == medium2 else 0.0
            similarity_scores.append(('medium', medium_similarity, 0.2))
        
        # Color temperature similarity (0.3 weight)
        color1 = artwork1.get('color', {})
        color2 = artwork2.get('color', {})
        temp1 = color1.get('temperature', '')
        temp2 = color2.get('temperature', '')
        if temp1 and temp2:
            temp_similarity = 1.0 if temp1 == temp2 else 0.0
            similarity_scores.append(('color_temp', temp_similarity, 0.3))
        
        # Calculate weighted average
        if similarity_scores:
            total_weighted_score = sum(score * weight for _, score, weight in similarity_scores)
            total_weight = sum(weight for _, _, weight in similarity_scores)
            return total_weighted_score / total_weight
        else:
            # No features to compare, assume low similarity
            return 0.2
    
    def _add_justifications(self, 
                          recommendations: List[ScoringResult], 
                          rag_brief: Dict[str, Any]) -> None:
        """Add justification text to final recommendations using LLM with caching"""
        
        logger.info(f"Generating justifications for {len(recommendations)} recommendations...")
        brief_hash = self._hash_brief(rag_brief)
        
        justification_cache_hits = 0
        
        for i, rec in enumerate(recommendations):
            try:
                # Check cache first
                cache_key = f"{brief_hash}_{rec.artwork_id}_justification"
                
                if cache_key in self.justification_cache:
                    # Use cached justification
                    rec.justification = self.justification_cache[cache_key]['justification']
                    justification_cache_hits += 1
                    logger.info(f"Using cached justification {i+1}/{len(recommendations)}")
                    continue
                
                # Generate new justification
                rec_dict = rec.to_dict()
                prompt = Step6Prompts.get_justification_prompt(rec_dict, rag_brief)
                
                # Call LLM
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Clean and assign justification
                justification = response.content.strip()
                
                # Remove any markdown formatting or extra quotes
                if justification.startswith('"') and justification.endswith('"'):
                    justification = justification[1:-1]
                
                rec.justification = justification
                
                # Cache the justification
                self.justification_cache[cache_key] = {
                    'justification': justification,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Generated justification {i+1}/{len(recommendations)}")
                
            except Exception as e:
                logger.error(f"Failed to generate justification for {rec.artwork_id}: {e}")
                # Fallback justification
                fallback_justification = f"Recommended based on strong therapeutic potential (score: {rec.weighted_score:.2f}/1.0) with particular strength in emotional support and evidence-based color psychology."
                rec.justification = fallback_justification
                
                # Cache fallback too (to avoid repeated failures)
                cache_key = f"{brief_hash}_{rec.artwork_id}_justification"
                self.justification_cache[cache_key] = {
                    'justification': fallback_justification,
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info(f"Completed justifications for {len(recommendations)} recommendations (cache hits: {justification_cache_hits})")


def main():
    """Test the Core Reranker with mock data"""
    
    # Mock RAG brief
    mock_brief = {
        "situation_analysis": "User experiencing work-related stress, needs calming environment",
        "curation_strategy": "Select visually soothing artworks with cool colors and peaceful themes",
        "curatorial_goals": ["Reduce stress", "Enhance focus", "Create calming environment"],
        "visual_elements": {
            "preferred_themes": ["landscapes", "abstract calm", "nature"],
            "color_psychology": {
                "primary_hues": ["soft blues", "gentle greens"],
                "color_temperature": "cool colors preferred"
            }
        }
    }
    
    # Mock candidates with diversity for MMR testing
    mock_candidates = [
        {
            "artwork_id": "aic_12345",
            "title": "Serene Lake Landscape",
            "artist_display": "Monet, Claude",
            "date_display": "1890",
            "medium_display": "Oil on canvas",
            "description": "A peaceful lake with soft blue and green reflections, perfect for stress relief.",
            "style_titles": ["Impressionism", "Landscape"],
            "subject_ids": ["nature", "landscape", "water"],
            "color": {"temperature": "cool", "dominant": "blue"},
            "score": 0.85,
            "source": "A1_metadata"
        },
        {
            "artwork_id": "aic_12346", 
            "title": "Abstract Flow",
            "artist_display": "Kandinsky, Wassily",
            "date_display": "1925",
            "medium_display": "Watercolor on paper",
            "description": "Abstract composition with flowing lines and gentle curves in calming tones.",
            "style_titles": ["Abstract", "Expressionism"],
            "subject_ids": ["abstract", "geometric"],
            "color": {"temperature": "neutral", "dominant": "purple"},
            "score": 0.75,
            "source": "A2_clip"
        },
        {
            "artwork_id": "aic_12347",
            "title": "Mountain Vista",
            "artist_display": "Bierstadt, Albert", 
            "date_display": "1860",
            "medium_display": "Oil on canvas",
            "description": "Majestic mountain landscape with dramatic lighting and natural beauty.",
            "style_titles": ["Realism", "Landscape"],
            "subject_ids": ["nature", "landscape", "mountains"],
            "color": {"temperature": "warm", "dominant": "orange"},
            "score": 0.70,
            "source": "A1_metadata"
        },
        {
            "artwork_id": "aic_12348",
            "title": "Portrait Study",
            "artist_display": "Renoir, Pierre-Auguste",
            "date_display": "1875", 
            "medium_display": "Oil on canvas",
            "description": "Gentle portrait with soft lighting and warm human expression.",
            "style_titles": ["Impressionism", "Portrait"],
            "subject_ids": ["portrait", "figure"],
            "color": {"temperature": "warm", "dominant": "pink"},
            "score": 0.65,
            "source": "A2_clip"
        },
        {
            "artwork_id": "aic_12349",
            "title": "Minimalist Composition",
            "artist_display": "Rothko, Mark",
            "date_display": "1958",
            "medium_display": "Acrylic on canvas", 
            "description": "Large color field painting with subtle tonal variations promoting contemplation.",
            "style_titles": ["Abstract", "Minimalism"],
            "subject_ids": ["abstract", "color"],
            "color": {"temperature": "cool", "dominant": "blue"},
            "score": 0.80,
            "source": "A2_clip"
        },
        {
            "artwork_id": "aic_12350",
            "title": "Forest Path",
            "artist_display": "Corot, Jean-Baptiste",
            "date_display": "1840",
            "medium_display": "Oil on canvas",
            "description": "Quiet forest scene with dappled light filtering through trees.",
            "style_titles": ["Realism", "Landscape"],
            "subject_ids": ["nature", "landscape", "trees"],
            "color": {"temperature": "cool", "dominant": "green"},
            "score": 0.78,
            "source": "A1_metadata"
        },
        {
            "artwork_id": "aic_12351",
            "title": "Geometric Study",
            "artist_display": "Mondrian, Piet",
            "date_display": "1920",
            "medium_display": "Oil on canvas",
            "description": "Clean geometric composition with primary colors and balanced structure.",
            "style_titles": ["Abstract", "Geometric"],
            "subject_ids": ["abstract", "geometric"],
            "color": {"temperature": "neutral", "dominant": "red"},
            "score": 0.60,
            "source": "A2_clip"
        },
        {
            "artwork_id": "aic_12352",
            "title": "Ocean Waves",
            "artist_display": "Hokusai, Katsushika",
            "date_display": "1830", 
            "medium_display": "Woodblock print",
            "description": "Dynamic ocean scene with rhythmic wave patterns in traditional style.",
            "style_titles": ["Ukiyo-e", "Landscape"],
            "subject_ids": ["nature", "water", "waves"],
            "color": {"temperature": "cool", "dominant": "blue"},
            "score": 0.72,
            "source": "A1_metadata"
        }
    ]
    
    # Initialize reranker
    reranker = Step6LLMReranker()
    
    # Test reranking
    try:
        result = reranker.rerank_candidates(mock_brief, mock_candidates, target_count=4)
        
        print("\n=== Step 6 Reranking Results ===")
        print(f"Input candidates: {result['metadata']['input_count']}")
        print(f"Final recommendations: {result['metadata']['output_count']}")
        print(f"Processing time: {result['metadata']['processing_time_seconds']:.2f}s")
        print(f"Cache hits: {result['metadata']['cache_hits']}")
        
        print("\nTop 5 Recommendations:")
        for i, rec in enumerate(result['final_recommendations'][:5]):
            print(f"{i+1}. {rec['artwork_id']} (score: {rec['rerank_score']:.3f})")
            print(f"   Breakdown: emotional={rec['scoring_breakdown']['emotional_fit']:.2f}, "
                  f"narrative={rec['scoring_breakdown']['narrative_fit']:.2f}, "
                  f"palette={rec['scoring_breakdown']['palette_fit']:.2f}")
        
    except Exception as e:
        print(f"Error testing reranker: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()