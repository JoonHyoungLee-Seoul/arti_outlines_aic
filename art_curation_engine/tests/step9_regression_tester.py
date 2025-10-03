#!/usr/bin/env python3
"""
Step 9: Regression Testing Pipeline for Art Recommendation System

This module provides automated regression testing capabilities for the complete
Step 5 → Stage A → Step 6 pipeline, including baseline comparison and 
performance monitoring.
"""

import json
import time
import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import pipeline components
try:
    from rag_session_langchain import RAGSessionBrief
    from stage_a_candidate_collection import StageACollector  
    from step6_llm_reranking import Step6LLMReranking
except ImportError as e:
    logging.warning(f"Could not import pipeline components: {e}")
    logging.info("Running in demo mode with mock components")

# Import quality validator components
from .step9_quality_validator import PipelineValidator, ScenarioResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineManager:
    """Manages baseline results for regression testing"""
    
    def __init__(self, baseline_dir: str = "tests/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        
    def save_baseline(self, scenario_id: str, result: ScenarioResult) -> None:
        """Save a test result as baseline for future regression testing"""
        baseline_file = self.baseline_dir / f"{scenario_id}_baseline.json"
        
        baseline_data = {
            "scenario_id": scenario_id,
            "timestamp": result.timestamp,
            "overall_score": result.overall_score,
            "processing_time": result.processing_time,
            "validation_results": [
                {
                    "check_name": vr.check_name,
                    "score": vr.score,
                    "actual_value": vr.actual_value
                } for vr in result.validation_results
            ],
            "pipeline_metrics": {
                "evidence_count": self._extract_metric(result, "step5_result.evidence_count"),
                "stage_a_candidates": self._extract_metric(result, "stage_a_result.final_candidates"),
                "final_recommendations": self._extract_metric(result, "step6_result.final_recommendations_count"),
                "evidence_alignment": self._extract_evidence_alignment(result)
            }
        }
        
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Baseline saved for scenario {scenario_id}")
    
    def load_baseline(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Load baseline data for a scenario"""
        baseline_file = self.baseline_dir / f"{scenario_id}_baseline.json"
        
        if not baseline_file.exists():
            logger.warning(f"No baseline found for scenario {scenario_id}")
            return None
        
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load baseline for {scenario_id}: {e}")
            return None
    
    def compare_with_baseline(self, scenario_id: str, current_result: ScenarioResult) -> Dict[str, Any]:
        """Compare current result with baseline"""
        baseline = self.load_baseline(scenario_id)
        
        if not baseline:
            return {
                "has_baseline": False,
                "regression_detected": False,
                "message": "No baseline available for comparison"
            }
        
        # Compare key metrics
        current_score = current_result.overall_score
        baseline_score = baseline["overall_score"]
        score_regression = (current_score - baseline_score) / baseline_score if baseline_score > 0 else 0
        
        current_time = current_result.processing_time
        baseline_time = baseline["processing_time"]
        time_regression = (current_time - baseline_time) / baseline_time if baseline_time > 0 else 0
        
        # Detection thresholds
        score_threshold = -0.05  # 5% decrease in score
        time_threshold = 0.50    # 50% increase in time
        
        regressions = []
        
        if score_regression < score_threshold:
            regressions.append(f"Quality regression: {score_regression:.2%} score decrease")
        
        if time_regression > time_threshold:
            regressions.append(f"Performance regression: {time_regression:.2%} time increase")
        
        # Compare individual validation checks
        baseline_checks = {vr["check_name"]: vr["score"] for vr in baseline["validation_results"]}
        current_checks = {vr.check_name: vr.score for vr in current_result.validation_results}
        
        for check_name, current_check_score in current_checks.items():
            if check_name in baseline_checks:
                baseline_check_score = baseline_checks[check_name]
                check_regression = (current_check_score - baseline_check_score) / baseline_check_score if baseline_check_score > 0 else 0
                
                if check_regression < -0.10:  # 10% decrease threshold for individual checks
                    regressions.append(f"{check_name}: {check_regression:.2%} decrease")
        
        return {
            "has_baseline": True,
            "baseline_timestamp": baseline["timestamp"],
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "score_change": score_regression,
            "time_change": time_regression,
            "baseline_score": baseline_score,
            "current_score": current_score,
            "baseline_time": baseline_time,
            "current_time": current_time
        }
    
    def _extract_metric(self, result: ScenarioResult, path: str) -> Any:
        """Extract metric from nested pipeline output"""
        keys = path.split('.')
        value = result.pipeline_output
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _extract_evidence_alignment(self, result: ScenarioResult) -> Optional[float]:
        """Extract evidence alignment score from validation results"""
        for vr in result.validation_results:
            if vr.check_name == "evidence_alignment":
                return vr.actual_value
        return None

class PipelineExecutor:
    """Executes the complete pipeline for testing"""
    
    def __init__(self):
        self.components_available = self._check_components()
        
    def _check_components(self) -> bool:
        """Check if all pipeline components are available"""
        try:
            # Test imports
            from rag_session_langchain import RAGSessionBrief
            from stage_a_candidate_collection import StageACollector
            from step6_llm_reranking import Step6LLMReranking
            return True
        except ImportError:
            return False
    
    def execute_pipeline(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete pipeline for a test scenario"""
        if not self.components_available:
            return self._mock_pipeline_execution(scenario)
        
        start_time = time.time()
        
        try:
            # Step 5: RAG Brief Generation
            step5_start = time.time()
            rag_session = RAGSessionBrief()
            brief_result = rag_session.generate_brief(
                scenario["situation"], 
                scenario["emotions"]
            )
            step5_time = time.time() - step5_start
            
            # Stage A: Candidate Collection
            stage_a_start = time.time()
            collector = StageACollector()
            candidates_result = collector.collect_candidates(
                scenario["situation"], 
                scenario["emotions"],
                mode="balanced"  # 150 candidates
            )
            stage_a_time = time.time() - stage_a_start
            
            # Step 6: LLM Reranking
            step6_start = time.time()
            reranker = Step6LLMReranking()
            final_result = reranker.rerank_candidates(
                brief_result["curation_brief"],
                brief_result["evidence_used"],
                candidates_result["final_candidates"]
            )
            step6_time = time.time() - step6_start
            
            total_time = time.time() - start_time
            
            return {
                "step5_result": {
                    "time": step5_time,
                    "evidence_count": len(brief_result.get("evidence_used", [])),
                    "brief_generated": True
                },
                "stage_a_result": {
                    "time": stage_a_time,
                    "final_candidates": len(candidates_result.get("final_candidates", [])),
                    "A1_metadata_hits": candidates_result.get("debug", {}).get("A1_hits", 0),
                    "A2_clip_hits": candidates_result.get("debug", {}).get("A2_hits", 0)
                },
                "step6_result": {
                    "time": step6_time,
                    "final_recommendations": final_result
                },
                "total_time": total_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "error": str(e),
                "total_time": time.time() - start_time,
                "success": False
            }
    
    def _mock_pipeline_execution(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Mock pipeline execution for testing when components are not available"""
        logger.info("Running mock pipeline execution")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate mock results based on scenario
        mock_recommendations = []
        for i in range(30):
            mock_recommendations.append({
                "artwork_id": 27000 + i,
                "llm_score": 0.85 + (i * -0.01),  # Decreasing scores
                "scores": {
                    "emotional_fit": 0.88 + (i * -0.005),
                    "narrative_fit": 0.83 + (i * -0.005),
                    "subject_fit": 0.87 + (i * -0.005),
                    "palette_fit": 0.82 + (i * -0.005),
                    "style_fit": 0.85 + (i * -0.005),
                    "evidence_alignment": 0.89 + (i * -0.005)
                },
                "reasoning": f"Mock recommendation {i+1} with evidence-based rationale...",
                "evidence_used": ["Mock study reference"]
            })
        
        return {
            "step5_result": {
                "time": 0.02,
                "evidence_count": 12,
                "brief_generated": True
            },
            "stage_a_result": {
                "time": 1.5,
                "final_candidates": 150,
                "A1_metadata_hits": 200,
                "A2_clip_hits": 150
            },
            "step6_result": {
                "time": 2.8,
                "final_recommendations": mock_recommendations
            },
            "total_time": 4.32,
            "success": True
        }

class RegressionTester:
    """Main regression testing coordinator"""
    
    def __init__(self, test_scenarios_path: str = "tests/test_scenarios.jsonl"):
        self.validator = PipelineValidator(test_scenarios_path)
        self.baseline_manager = BaselineManager()
        self.executor = PipelineExecutor()
        self.results_dir = Path("tests/results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_regression_tests(self, scenario_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run regression tests for specified scenario types"""
        scenarios = self.validator.load_test_scenarios()
        
        if scenario_types:
            scenarios = [s for s in scenarios if s['test_type'] in scenario_types]
        
        if not scenarios:
            logger.error("No scenarios to test")
            return {"error": "No scenarios found"}
        
        logger.info(f"Running regression tests on {len(scenarios)} scenarios")
        
        all_results = []
        regression_summary = {
            "total_scenarios": len(scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "regressions_detected": 0,
            "new_baselines_created": 0,
            "scenario_details": []
        }
        
        for scenario in scenarios:
            logger.info(f"Testing scenario: {scenario['scenario_id']}")
            
            # Execute pipeline
            pipeline_output = self.executor.execute_pipeline(scenario)
            
            if not pipeline_output.get("success", False):
                logger.error(f"Pipeline execution failed for {scenario['scenario_id']}")
                continue
            
            # Validate results
            validation_result = self.validator.validate_scenario(scenario, pipeline_output)
            all_results.append(validation_result)
            
            # Compare with baseline
            baseline_comparison = self.baseline_manager.compare_with_baseline(
                scenario['scenario_id'], validation_result
            )
            
            # Update summary
            if validation_result.passed:
                regression_summary["passed_scenarios"] += 1
            else:
                regression_summary["failed_scenarios"] += 1
            
            if baseline_comparison["regression_detected"]:
                regression_summary["regressions_detected"] += 1
            
            if not baseline_comparison["has_baseline"]:
                # Create new baseline
                self.baseline_manager.save_baseline(scenario['scenario_id'], validation_result)
                regression_summary["new_baselines_created"] += 1
            
            # Add to detailed results
            scenario_detail = {
                "scenario_id": scenario['scenario_id'],
                "test_type": scenario['test_type'],
                "passed": validation_result.passed,
                "overall_score": validation_result.overall_score,
                "processing_time": validation_result.processing_time,
                "baseline_comparison": baseline_comparison
            }
            regression_summary["scenario_details"].append(scenario_detail)
            
            logger.info(f"Scenario {scenario['scenario_id']}: "
                       f"{'PASS' if validation_result.passed else 'FAIL'} "
                       f"(score: {validation_result.overall_score:.3f})")
        
        # Generate comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"regression_report_{timestamp}.json"
        
        full_report = {
            "regression_summary": regression_summary,
            "detailed_test_report": self.validator.generate_test_report(
                all_results, 
                str(self.results_dir / f"detailed_report_{timestamp}.json")
            )
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Regression test report saved to {report_file}")
        self._print_regression_summary(regression_summary)
        
        return full_report
    
    def update_baselines(self, scenario_ids: Optional[List[str]] = None) -> None:
        """Update baselines for specified scenarios"""
        scenarios = self.validator.load_test_scenarios()
        
        if scenario_ids:
            scenarios = [s for s in scenarios if s['scenario_id'] in scenario_ids]
        
        logger.info(f"Updating baselines for {len(scenarios)} scenarios")
        
        for scenario in scenarios:
            logger.info(f"Updating baseline for: {scenario['scenario_id']}")
            
            pipeline_output = self.executor.execute_pipeline(scenario)
            
            if pipeline_output.get("success", False):
                validation_result = self.validator.validate_scenario(scenario, pipeline_output)
                self.baseline_manager.save_baseline(scenario['scenario_id'], validation_result)
            else:
                logger.error(f"Failed to update baseline for {scenario['scenario_id']}")
    
    def _print_regression_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted regression test summary"""
        print("\n" + "="*60)
        print("REGRESSION TEST SUMMARY")
        print("="*60)
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Passed: {summary['passed_scenarios']}")
        print(f"Failed: {summary['failed_scenarios']}")
        print(f"Pass Rate: {summary['passed_scenarios']/summary['total_scenarios']:.1%}")
        print(f"Regressions Detected: {summary['regressions_detected']}")
        print(f"New Baselines Created: {summary['new_baselines_created']}")
        
        if summary['regressions_detected'] > 0:
            print("\n⚠️  REGRESSIONS DETECTED:")
            for detail in summary['scenario_details']:
                if detail['baseline_comparison'].get('regression_detected', False):
                    print(f"  - {detail['scenario_id']}: {detail['baseline_comparison']['regressions']}")
        
        print("\n" + "="*60)

def main():
    """Main entry point for regression testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Art Recommendation Pipeline Regression Tester")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenario IDs to test")
    parser.add_argument("--types", nargs="+", choices=["regression", "quality", "edge_case"], 
                       help="Test types to run")
    parser.add_argument("--update-baselines", action="store_true", 
                       help="Update baselines instead of running tests")
    
    args = parser.parse_args()
    
    tester = RegressionTester()
    
    if args.update_baselines:
        tester.update_baselines(args.scenarios)
    else:
        tester.run_regression_tests(args.types)

if __name__ == "__main__":
    main()