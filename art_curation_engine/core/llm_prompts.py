#!/usr/bin/env python3

"""
LLM Prompts for Step 6 Reranking System

Contains optimized prompts for:
- Multi-dimensional artwork scoring (6 dimensions)
- Batch scoring efficiency
- Justification generation
- Error handling prompts
"""

from typing import Dict, List, Any
import json

class Step6Prompts:
    """Collection of prompts for Step 6 LLM reranking"""
    
    @staticmethod
    def get_batch_scoring_prompt(rag_brief: Dict[str, Any], 
                               candidates: List[Dict[str, Any]]) -> str:
        """
        Generate prompt for batch scoring of artwork candidates
        
        Args:
            rag_brief: Step 5 generated brief with situation analysis
            candidates: List of artwork candidates to score
            
        Returns:
            Formatted prompt string for LLM scoring
        """
        
        # Extract key elements from RAG brief
        situation = rag_brief.get('situation_analysis', 'No situation provided')
        strategy = rag_brief.get('curation_strategy', 'No strategy provided')
        goals = rag_brief.get('curatorial_goals', [])
        visual_elements = rag_brief.get('visual_elements', {})
        
        # Format preferred themes and colors
        preferred_themes = visual_elements.get('preferred_themes', [])
        color_psychology = visual_elements.get('color_psychology', {})
        primary_hues = color_psychology.get('primary_hues', [])
        color_temp = color_psychology.get('color_temperature', 'Not specified')
        
        # Format artwork list for scoring
        artwork_list = []
        for i, candidate in enumerate(candidates, 1):
            artwork_info = {
                'id': candidate.get('artwork_id', f'unknown_{i}'),
                'title': candidate.get('title', 'Untitled'),
                'artist': candidate.get('artist_display', 'Unknown Artist'),
                'date': candidate.get('date_display', 'Unknown Date'),
                'medium': candidate.get('medium_display', 'Unknown Medium'),
                'description': candidate.get('description', 'No description available'),
                'style': candidate.get('style_titles', []),
                'subject_matter': candidate.get('subject_ids', []),
                'color_info': candidate.get('color', {}),
                'source': candidate.get('source', 'unknown')
            }
            
            artwork_text = f"""
{i}. ID: {artwork_info['id']}
   Title: {artwork_info['title']}
   Artist: {artwork_info['artist']}
   Date: {artwork_info['date']}
   Medium: {artwork_info['medium']}
   Style: {', '.join(artwork_info['style']) if artwork_info['style'] else 'Not specified'}
   Subject: {', '.join(map(str, artwork_info['subject_matter'])) if artwork_info['subject_matter'] else 'Not specified'}
   Description: {artwork_info['description'][:200]}...
   Source: {artwork_info['source']}
"""
            artwork_list.append(artwork_text.strip())
        
        prompt = f"""You are an expert art therapist and color psychology specialist. Your task is to evaluate artworks for their therapeutic suitability based on a specific user situation and evidence-based recommendations.

## SITUATION ANALYSIS
{situation}

## CURATION STRATEGY  
{strategy}

## CURATORIAL GOALS
{chr(10).join(f"- {goal}" for goal in goals)}

## VISUAL REQUIREMENTS
**Preferred Themes:** {', '.join(preferred_themes) if preferred_themes else 'Not specified'}
**Recommended Colors:** {', '.join(primary_hues) if primary_hues else 'Not specified'}
**Color Temperature:** {color_temp}

## ARTWORKS TO EVALUATE
{chr(10).join(artwork_list)}

## SCORING INSTRUCTIONS
Evaluate each artwork on these 6 dimensions using a 0.0-1.0 scale:

1. **emotional_fit** (0.0-1.0): How well does this artwork support the user's emotional needs?
   - Consider color psychology effects on mood
   - Assess thematic resonance with emotional state
   - Evaluate potential for emotional regulation

2. **narrative_fit** (0.0-1.0): How well does this artwork align with the user's life situation?
   - Consider contextual appropriateness 
   - Assess symbolic relevance to circumstances
   - Evaluate narrative coherence with user's story

3. **subject_fit** (0.0-1.0): How well does the subject matter match preferences?
   - Match against preferred themes
   - Consider subject matter appropriateness
   - Assess visual content suitability

4. **palette_fit** (0.0-1.0): How well does the color palette serve therapeutic goals?
   - Apply color psychology principles
   - Match against recommended color temperature
   - Consider saturation and brightness effects

5. **style_fit** (0.0-1.0): How well does the artistic style support the therapeutic purpose?
   - Consider stylistic preferences mentioned
   - Assess visual complexity appropriateness
   - Evaluate artistic approach alignment

6. **evidence_alignment** (0.0-1.0): How well does this recommendation align with scientific evidence?
   - Consider research-backed color effects
   - Assess evidence-based therapeutic principles
   - Evaluate consistency with established art therapy practices

## OUTPUT FORMAT
Respond with ONLY a valid JSON object containing scores for each artwork:

```json
{{
  "scores": [
    {{
      "artwork_id": "artwork_id_here",
      "emotional_fit": 0.85,
      "narrative_fit": 0.72,
      "subject_fit": 0.90,
      "palette_fit": 0.78,
      "style_fit": 0.65,
      "evidence_alignment": 0.80
    }}
  ]
}}
```

IMPORTANT: 
- Provide exactly one score object for each artwork in the same order
- Use only numeric values between 0.0 and 1.0
- Consider the therapeutic context in all evaluations
- Be conservative with very high scores (>0.9) - reserve for exceptional matches
- Focus on evidence-based recommendations over subjective preferences"""

        return prompt
    
    @staticmethod
    def get_justification_prompt(recommendation: Dict[str, Any], 
                               rag_brief: Dict[str, Any]) -> str:
        """
        Generate prompt for creating recommendation justifications
        
        Args:
            recommendation: Single artwork recommendation with scores
            rag_brief: Step 5 generated brief
            
        Returns:
            Formatted prompt for justification generation
        """
        
        situation = rag_brief.get('situation_analysis', 'No situation provided')
        strategy = rag_brief.get('curation_strategy', 'No strategy provided')
        
        scores = recommendation.get('scoring_breakdown', {})
        artwork_id = recommendation.get('artwork_id', 'unknown')
        overall_score = recommendation.get('rerank_score', 0.0)
        
        prompt = f"""You are an art therapist explaining your artwork recommendation to a client. Provide a clear, professional justification for why this specific artwork is recommended.

## CLIENT SITUATION
{situation}

## THERAPEUTIC STRATEGY
{strategy}

## ARTWORK RECOMMENDATION
**Artwork ID:** {artwork_id}
**Overall Suitability:** {overall_score:.2f}/1.0

**Detailed Assessment:**
- Emotional Support: {scores.get('emotional_fit', 0):.2f}/1.0
- Situational Relevance: {scores.get('narrative_fit', 0):.2f}/1.0  
- Subject Appropriateness: {scores.get('subject_fit', 0):.2f}/1.0
- Color Psychology: {scores.get('palette_fit', 0):.2f}/1.0
- Artistic Style: {scores.get('style_fit', 0):.2f}/1.0
- Evidence Base: {scores.get('evidence_alignment', 0):.2f}/1.0

## TASK
Write a concise, professional justification (2-3 sentences) explaining:
1. Why this artwork is particularly suitable for this client
2. Which specific therapeutic benefits it may provide
3. How it aligns with evidence-based art therapy principles

Use warm, supportive language appropriate for a therapeutic context. Focus on the strongest scoring dimensions but mention any notable therapeutic considerations.

Respond with ONLY the justification text (no JSON, no additional formatting)."""

        return prompt
    
    @staticmethod 
    def get_fallback_scoring_prompt(candidates: List[Dict[str, Any]]) -> str:
        """
        Generate simplified prompt when main LLM scoring fails
        
        Args:
            candidates: List of artwork candidates
            
        Returns:
            Simplified scoring prompt
        """
        
        artwork_ids = [c.get('artwork_id', 'unknown') for c in candidates]
        
        prompt = f"""Rate these artworks for general therapeutic suitability on a 0.0-1.0 scale.

Artworks: {', '.join(artwork_ids)}

Consider basic factors:
- Calming vs stimulating visual qualities
- Positive vs negative emotional associations  
- Simple vs complex compositions
- Warm vs cool color temperatures

Respond with JSON format:
```json
{{
  "scores": [
    {{
      "artwork_id": "id_here", 
      "overall_score": 0.75
    }}
  ]
}}
```"""
        
        return prompt

    @staticmethod
    def get_diversity_prompt(high_scoring_candidates: List[Dict[str, Any]], 
                           target_count: int) -> str:
        """
        Generate prompt for MMR diversity selection
        
        Args:
            high_scoring_candidates: Top scoring candidates
            target_count: Number of diverse selections needed
            
        Returns:
            Diversity selection prompt  
        """
        
        candidate_info = []
        for candidate in high_scoring_candidates:
            info = {
                'id': candidate.get('artwork_id', 'unknown'),
                'score': candidate.get('rerank_score', 0.0),
                'style': candidate.get('style_titles', []),
                'subject': candidate.get('subject_ids', []),
                'medium': candidate.get('medium_display', 'Unknown'),
                'color_temp': candidate.get('color', {}).get('temperature', 'Unknown')
            }
            candidate_info.append(info)
        
        prompt = f"""Select {target_count} artworks from these high-scoring candidates to maximize diversity while maintaining quality.

CANDIDATES:
{json.dumps(candidate_info, indent=2)}

DIVERSITY CRITERIA:
- Vary artistic styles (avoid too many similar styles)
- Vary subject matter (landscapes, portraits, abstracts, etc.)
- Vary color temperatures (mix warm/cool as appropriate)
- Vary mediums (paintings, sculptures, prints, etc.)
- Maintain high therapeutic scores

Select artworks that together provide a diverse, well-rounded collection while keeping the highest-scoring pieces.

Respond with JSON:
```json
{{
  "selected_ids": ["id1", "id2", "id3", ...]
}}
```"""
        
        return prompt

def main():
    """Test prompt generation"""
    
    # Test scoring prompt
    mock_brief = {
        "situation_analysis": "User experiencing work stress, needs calming environment",
        "curation_strategy": "Select visually soothing artworks with cool colors",
        "curatorial_goals": ["Reduce stress", "Enhance focus"],
        "visual_elements": {
            "preferred_themes": ["landscapes", "abstract calm"],
            "color_psychology": {
                "primary_hues": ["soft blues", "gentle greens"],
                "color_temperature": "cool colors preferred"
            }
        }
    }
    
    mock_candidates = [
        {
            "artwork_id": "aic_12345",
            "title": "Peaceful Lake",
            "artist_display": "Unknown Artist",
            "medium_display": "Oil on canvas",
            "description": "A serene lake scene with soft blue tones...",
            "source": "A1_metadata"
        }
    ]
    
    # Generate and print scoring prompt
    scoring_prompt = Step6Prompts.get_batch_scoring_prompt(mock_brief, mock_candidates)
    print("=== SCORING PROMPT SAMPLE ===")
    print(scoring_prompt[:1000] + "...\n")
    
    # Test justification prompt
    mock_rec = {
        "artwork_id": "aic_12345",
        "rerank_score": 0.85,
        "scoring_breakdown": {
            "emotional_fit": 0.90,
            "narrative_fit": 0.80,
            "palette_fit": 0.85,
            "subject_fit": 0.88,
            "style_fit": 0.75,
            "evidence_alignment": 0.82
        }
    }
    
    justification_prompt = Step6Prompts.get_justification_prompt(mock_rec, mock_brief)
    print("=== JUSTIFICATION PROMPT SAMPLE ===")
    print(justification_prompt[:500] + "...\n")

if __name__ == "__main__":
    main()