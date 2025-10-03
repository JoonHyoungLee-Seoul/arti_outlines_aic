#!/usr/bin/env python3

"""
End-to-End Integration Test: Step 5 → Stage A → Step 6

Tests the complete pipeline from user input to final 30 recommendations
with comprehensive validation and performance monitoring.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Set cache directories before imports
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

from dotenv import load_dotenv
load_dotenv(override=True)

# Import our systems
from core.rag_session_langchain import RAGSessionBrief
from core.stage_a_candidate_collection import StageACollector
from core.step6_llm_reranking import Step6LLMReranker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EndToEndTester:
    """Complete pipeline tester for Step 5 → Stage A → Step 6"""
    
    def __init__(self):
        """Initialize all pipeline components"""
        logger.info("Initializing End-to-End Pipeline Tester...")
        
        # Initialize components
        self.rag_session = RAGSessionBrief()
        self.stage_a_collector = StageACollector()
        self.step6_reranker = Step6LLMReranker(batch_size=10, max_workers=2)  # Smaller batches for testing
        
        logger.info("All pipeline components initialized successfully")
    
    def run_complete_pipeline(self, situation: str, emotions: List[str], test_name: str = "test") -> Dict[str, Any]:
        """
        Run the complete Step 5 → Stage A → Step 6 pipeline
        
        Args:
            situation: User's situation description (pre-analyzed by LangGraph)
            emotions: List of user emotions (pre-extracted by LangGraph)
            test_name: Name for this test run
            
        Returns:
            Complete pipeline results with performance metrics
        """
        
        logger.info(f"=== Starting End-to-End Test: {test_name} ===")
        start_time = time.time()
        
        pipeline_result = {
            'test_name': test_name,
            'input': {
                'situation': situation,
                'emotions': emotions
            },
            'timestamp': datetime.now().isoformat(),
            'step5_result': None,
            'stage_a_result': None,
            'step6_result': None,
            'performance': {},
            'validation': {}
        }
        
        try:
            # === STEP 5: RAG Brief Generation ===
            logger.info("Phase 1: Step 5 RAG Brief Generation...")
            step5_start = time.time()
            
            # Use pre-processed situation and emotions from LangGraph
            rag_brief = self.rag_session.generate_brief(situation, emotions)
            
            step5_time = time.time() - step5_start
            pipeline_result['step5_result'] = rag_brief
            pipeline_result['performance']['step5_time'] = step5_time
            
            logger.info(f"Step 5 completed in {step5_time:.2f}s")
            
            # === STAGE A: Candidate Collection ===
            logger.info("Phase 2: Stage A Candidate Collection...")
            stage_a_start = time.time()
            
            # Use the pre-processed situation for Stage A
            stage_a_result = self.stage_a_collector.collect_candidates(situation, emotions)
            
            stage_a_time = time.time() - stage_a_start
            pipeline_result['stage_a_result'] = stage_a_result
            pipeline_result['performance']['stage_a_time'] = stage_a_time
            
            logger.info(f"Stage A completed in {stage_a_time:.2f}s - {len(stage_a_result.get('candidates', []))} candidates")
            
            # === STEP 6: LLM Reranking ===
            logger.info("Phase 3: Step 6 LLM Reranking...")
            step6_start = time.time()
            
            candidates = stage_a_result.get('final_stageA_ids', [])
            if len(candidates) < 30:
                logger.warning(f"Only {len(candidates)} candidates from Stage A, expecting ~150")
            
            # Convert artwork IDs to candidate objects with metadata
            candidate_objects = []
            if hasattr(self.stage_a_collector, 'metadata_index'):
                for artwork_id in candidates:
                    if artwork_id in self.stage_a_collector.metadata_index:
                        candidate_objects.append(self.stage_a_collector.metadata_index[artwork_id])
                    else:
                        # Fallback if metadata not found
                        candidate_objects.append({"id": artwork_id})
            else:
                # Fallback to ID-only objects
                candidate_objects = [{"id": artwork_id} for artwork_id in candidates]
            
            # Target 30 final recommendations or all candidates if fewer
            target_count = min(30, len(candidate_objects))
            step6_result = self.step6_reranker.rerank_candidates(rag_brief, candidate_objects, target_count)
            
            step6_time = time.time() - step6_start  
            pipeline_result['step6_result'] = step6_result
            pipeline_result['performance']['step6_time'] = step6_time
            
            logger.info(f"Step 6 completed in {step6_time:.2f}s - {len(step6_result.get('final_recommendations', []))} final recommendations")
            
            # === VALIDATION ===
            logger.info("Phase 4: Results Validation...")
            pipeline_result['validation'] = self._validate_pipeline_results(pipeline_result)
            
            # === TOTAL PERFORMANCE ===
            total_time = time.time() - start_time
            pipeline_result['performance']['total_time'] = total_time
            pipeline_result['performance']['pipeline_success'] = True
            
            logger.info(f"=== Pipeline Completed Successfully in {total_time:.2f}s ===")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            total_time = time.time() - start_time
            pipeline_result['performance']['total_time'] = total_time
            pipeline_result['performance']['pipeline_success'] = False
            pipeline_result['error'] = str(e)
            
            return pipeline_result
    
    def _validate_pipeline_results(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the pipeline results for correctness"""
        
        validation = {
            'step5_valid': False,
            'stage_a_valid': False,
            'step6_valid': False,
            'end_to_end_valid': False,
            'issues': []
        }
        
        # Validate Step 5
        step5_result = pipeline_result.get('step5_result')
        if step5_result and isinstance(step5_result, dict):
            required_keys = ['situation_analysis', 'curation_strategy', 'visual_elements']
            if all(key in step5_result for key in required_keys):
                validation['step5_valid'] = True
            else:
                missing = [key for key in required_keys if key not in step5_result]
                validation['issues'].append(f"Step 5 missing keys: {missing}")
        else:
            validation['issues'].append("Step 5 result invalid or missing")
        
        # Validate Stage A
        stage_a_result = pipeline_result.get('stage_a_result')
        if stage_a_result and isinstance(stage_a_result, dict):
            candidates = stage_a_result.get('final_stageA_ids', [])
            if isinstance(candidates, list) and len(candidates) > 0:
                validation['stage_a_valid'] = True
                validation['stage_a_candidate_count'] = len(candidates)
            else:
                validation['issues'].append(f"Stage A returned {len(candidates)} candidates")
        else:
            validation['issues'].append("Stage A result invalid or missing")
        
        # Validate Step 6
        step6_result = pipeline_result.get('step6_result')
        if step6_result and isinstance(step6_result, dict):
            recommendations = step6_result.get('final_recommendations', [])
            if isinstance(recommendations, list) and len(recommendations) > 0:
                validation['step6_valid'] = True
                validation['step6_recommendation_count'] = len(recommendations)
                
                # Check recommendation structure
                if recommendations:
                    sample_rec = recommendations[0]
                    required_rec_keys = ['artwork_id', 'rerank_score', 'scoring_breakdown', 'justification']
                    if all(key in sample_rec for key in required_rec_keys):
                        validation['step6_structure_valid'] = True
                    else:
                        validation['issues'].append("Step 6 recommendation structure invalid")
            else:
                validation['issues'].append(f"Step 6 returned {len(recommendations)} recommendations")
        else:
            validation['issues'].append("Step 6 result invalid or missing")
        
        # Overall validation
        validation['end_to_end_valid'] = (
            validation['step5_valid'] and 
            validation['stage_a_valid'] and 
            validation['step6_valid']
        )
        
        return validation
    
    def run_test_scenarios(self) -> List[Dict[str, Any]]:
        """Run multiple test scenarios to validate system robustness"""
        
        test_scenarios = [
            {
                'name': 'work_stress_scenario',
                'situation': 'work stress with concentration difficulties',
                'emotions': ['stress', 'anxiety', 'overwhelmed']
            },
            {
                'name': 'creative_inspiration_scenario', 
                'situation': 'creative project requiring visual inspiration',
                'emotions': ['curiosity', 'motivation', 'creative_energy']
            },
            {
                'name': 'evening_relaxation_scenario',
                'situation': 'end of day relaxation and unwinding',
                'emotions': ['tired', 'peaceful', 'contemplative']
            },
            {
                'name': 'mood_boost_scenario',
                'situation': 'feeling down and needing emotional uplift',
                'emotions': ['sadness', 'low_energy', 'hopeful']
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Test Scenario: {scenario['name']}")
            logger.info(f"Situation: {scenario['situation']}")
            logger.info(f"Emotions: {scenario['emotions']}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.run_complete_pipeline(scenario['situation'], scenario['emotions'], scenario['name'])
                results.append(result)
                
                # Log summary
                perf = result.get('performance', {})
                val = result.get('validation', {})
                
                logger.info(f"\nScenario Results:")
                logger.info(f"  Success: {perf.get('pipeline_success', False)}")
                logger.info(f"  Total Time: {perf.get('total_time', 0):.2f}s")
                logger.info(f"  Step 5: {perf.get('step5_time', 0):.2f}s")
                logger.info(f"  Stage A: {perf.get('stage_a_time', 0):.2f}s") 
                logger.info(f"  Step 6: {perf.get('step6_time', 0):.2f}s")
                logger.info(f"  Validation: {val.get('end_to_end_valid', False)}")
                if val.get('issues'):
                    logger.warning(f"  Issues: {val['issues']}")
                
            except Exception as e:
                logger.error(f"Scenario {scenario['name']} failed: {e}")
                results.append({
                    'test_name': scenario['name'],
                    'error': str(e),
                    'performance': {'pipeline_success': False}
                })
        
        return results
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'total_scenarios': len(results),
            'successful_scenarios': 0,
            'failed_scenarios': 0,
            'performance_summary': {
                'avg_total_time': 0,
                'avg_step5_time': 0,
                'avg_stage_a_time': 0,
                'avg_step6_time': 0
            },
            'validation_summary': {
                'step5_success_rate': 0,
                'stage_a_success_rate': 0,
                'step6_success_rate': 0,
                'end_to_end_success_rate': 0
            },
            'scenarios': results
        }
        
        # Calculate success rates
        successful = [r for r in results if r.get('performance', {}).get('pipeline_success', False)]
        report['successful_scenarios'] = len(successful)
        report['failed_scenarios'] = len(results) - len(successful)
        
        if successful:
            # Performance averages
            total_times = [r['performance']['total_time'] for r in successful]
            step5_times = [r['performance']['step5_time'] for r in successful]
            stage_a_times = [r['performance']['stage_a_time'] for r in successful]
            step6_times = [r['performance']['step6_time'] for r in successful]
            
            report['performance_summary']['avg_total_time'] = sum(total_times) / len(total_times)
            report['performance_summary']['avg_step5_time'] = sum(step5_times) / len(step5_times) 
            report['performance_summary']['avg_stage_a_time'] = sum(stage_a_times) / len(stage_a_times)
            report['performance_summary']['avg_step6_time'] = sum(step6_times) / len(step6_times)
            
            # Validation rates
            step5_valid = sum(1 for r in successful if r.get('validation', {}).get('step5_valid', False))
            stage_a_valid = sum(1 for r in successful if r.get('validation', {}).get('stage_a_valid', False))
            step6_valid = sum(1 for r in successful if r.get('validation', {}).get('step6_valid', False))
            end_to_end_valid = sum(1 for r in successful if r.get('validation', {}).get('end_to_end_valid', False))
            
            report['validation_summary']['step5_success_rate'] = step5_valid / len(successful)
            report['validation_summary']['stage_a_success_rate'] = stage_a_valid / len(successful)
            report['validation_summary']['step6_success_rate'] = step6_valid / len(successful)
            report['validation_summary']['end_to_end_success_rate'] = end_to_end_valid / len(successful)
        
        return report


def main():
    """Run comprehensive end-to-end testing"""
    
    print("🚀 Starting Comprehensive End-to-End Pipeline Testing")
    print("Testing: Step 5 → Stage A → Step 6 Integration")
    print("=" * 70)
    
    # Initialize tester
    tester = EndToEndTester()
    
    # Run test scenarios
    print("\n📋 Running Test Scenarios...")
    results = tester.run_test_scenarios()
    
    # Generate report
    print("\n📊 Generating Test Report...")
    report = tester.generate_test_report(results)
    
    # Save results
    os.makedirs("storage", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"storage/end_to_end_test_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🎯 END-TO-END PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    print(f"📈 Overall Success Rate: {report['successful_scenarios']}/{report['total_scenarios']} ({100*report['successful_scenarios']/report['total_scenarios']:.1f}%)")
    
    if report['successful_scenarios'] > 0:
        perf = report['performance_summary']
        val = report['validation_summary']
        
        print(f"\n⏱️  Average Performance:")
        print(f"   Total Pipeline: {perf['avg_total_time']:.2f}s")
        print(f"   Step 5 (RAG):   {perf['avg_step5_time']:.2f}s")
        print(f"   Stage A:        {perf['avg_stage_a_time']:.2f}s")
        print(f"   Step 6:         {perf['avg_step6_time']:.2f}s")
        
        print(f"\n✅ Component Success Rates:")
        print(f"   Step 5:         {100*val['step5_success_rate']:.1f}%")
        print(f"   Stage A:        {100*val['stage_a_success_rate']:.1f}%")
        print(f"   Step 6:         {100*val['step6_success_rate']:.1f}%")
        print(f"   End-to-End:     {100*val['end_to_end_success_rate']:.1f}%")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    if report['successful_scenarios'] == report['total_scenarios']:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready for production.")
    else:
        print(f"\n⚠️  {report['failed_scenarios']} tests failed. Check logs for details.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()