#!/usr/bin/env python3
"""
Step 9: Quality and Regression Testing System for Art Recommendation Pipeline

This module provides comprehensive quality validation and regression testing
for the Step 5 → Stage A → Step 6 pipeline.
"""

import json
import time
import hashlib
import statistics
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    actual_value: Any
    expected_range: Dict[str, Any]
    score: float  # 0.0 to 1.0
    message: str

@dataclass
class ScenarioResult:
    """Complete test result for a scenario"""
    scenario_id: str
    test_type: str
    passed: bool
    overall_score: float
    processing_time: float
    validation_results: List[ValidationResult]
    pipeline_output: Dict[str, Any]
    timestamp: str
    error_message: Optional[str] = None

class QualityMetrics:
    """Quality metrics calculator for pipeline output"""
    
    @staticmethod
    def calculate_evidence_alignment_score(final_recommendations: List[Dict]) -> float:
        """Calculate average evidence alignment score from final recommendations"""
        if not final_recommendations:
            return 0.0
        
        alignment_scores = []
        for rec in final_recommendations:
            if 'scores' in rec and 'evidence_alignment' in rec['scores']:
                alignment_scores.append(rec['scores']['evidence_alignment'])
        
        return statistics.mean(alignment_scores) if alignment_scores else 0.0
    
    @staticmethod
    def calculate_dimensional_balance(final_recommendations: List[Dict]) -> Dict[str, float]:
        """Calculate balance across 6 dimensional scores"""
        dimensions = ['emotional_fit', 'narrative_fit', 'subject_fit', 
                     'palette_fit', 'style_fit', 'evidence_alignment']
        
        dimension_scores = {dim: [] for dim in dimensions}
        
        for rec in final_recommendations:
            if 'scores' in rec:
                for dim in dimensions:
                    if dim in rec['scores']:
                        dimension_scores[dim].append(rec['scores'][dim])
        
        # Calculate average and standard deviation for each dimension
        balance_metrics = {}
        all_averages = []
        
        for dim, scores in dimension_scores.items():
            if scores:
                avg = statistics.mean(scores)
                balance_metrics[f'{dim}_avg'] = avg
                balance_metrics[f'{dim}_std'] = statistics.stdev(scores) if len(scores) > 1 else 0.0
                all_averages.append(avg)
        
        # Overall balance metrics
        if all_averages:
            balance_metrics['overall_avg'] = statistics.mean(all_averages)
            balance_metrics['overall_std'] = statistics.stdev(all_averages) if len(all_averages) > 1 else 0.0
        
        return balance_metrics
    
    @staticmethod
    def calculate_citation_coverage(final_recommendations: List[Dict]) -> float:
        """Calculate percentage of recommendations with evidence citations"""
        if not final_recommendations:
            return 0.0
        
        cited_count = 0
        for rec in final_recommendations:
            if 'evidence_used' in rec and rec['evidence_used']:
                cited_count += 1
            elif 'reasoning' in rec and any(keyword in rec['reasoning'].lower() 
                                          for keyword in ['study', 'research', 'et al', 'psychology']):
                cited_count += 1
        
        return cited_count / len(final_recommendations)
    
    @staticmethod
    def detect_theme_alignment(final_recommendations: List[Dict], 
                             expected_themes: List[str], 
                             avoid_themes: List[str]) -> Dict[str, float]:
        """Detect theme alignment in recommendations"""
        if not final_recommendations:
            return {"theme_match_rate": 0.0, "avoid_violation_rate": 0.0}
        
        theme_matches = 0
        avoid_violations = 0
        
        for rec in final_recommendations:
            reasoning = rec.get('reasoning', '').lower()
            
            # Check for expected themes
            if any(theme.lower() in reasoning for theme in expected_themes):
                theme_matches += 1
            
            # Check for avoided themes
            if any(theme.lower() in reasoning for theme in avoid_themes):
                avoid_violations += 1
        
        return {
            "theme_match_rate": theme_matches / len(final_recommendations),
            "avoid_violation_rate": avoid_violations / len(final_recommendations)
        }

class PipelineValidator:
    """Main validation engine for pipeline testing"""
    
    def __init__(self, test_scenarios_path: str = "tests/test_scenarios.jsonl"):
        self.test_scenarios_path = Path(test_scenarios_path)
        self.metrics = QualityMetrics()
        
    def load_test_scenarios(self) -> List[Dict[str, Any]]:
        """Load test scenarios from JSONL file"""
        scenarios = []
        try:
            with open(self.test_scenarios_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        scenarios.append(json.loads(line))
            logger.info(f"Loaded {len(scenarios)} test scenarios")
            return scenarios
        except Exception as e:
            logger.error(f"Failed to load test scenarios: {e}")
            return []
    
    def validate_scenario(self, scenario: Dict[str, Any], 
                         pipeline_output: Dict[str, Any]) -> ScenarioResult:
        """Validate a single scenario against pipeline output"""
        validation_results = []
        start_time = time.time()
        
        try:
            # Extract key components from pipeline output
            step5_result = pipeline_output.get('step5_result', {})
            stage_a_result = pipeline_output.get('stage_a_result', {})
            step6_result = pipeline_output.get('step6_result', {})
            final_recommendations = step6_result.get('final_recommendations', [])
            
            # Validate Step 5 evidence count
            evidence_count = step5_result.get('evidence_count', 0)
            expected_evidence = scenario['validation_checks']['step5_evidence_count']
            validation_results.append(self._validate_range(
                "step5_evidence_count", evidence_count, expected_evidence,
                f"Evidence count: {evidence_count}"
            ))
            
            # Validate Stage A candidate count
            stage_a_candidates = stage_a_result.get('final_candidates', 0)
            expected_candidates = scenario['validation_checks']['stage_a_candidates']
            validation_results.append(self._validate_range(
                "stage_a_candidates", stage_a_candidates, expected_candidates,
                f"Stage A candidates: {stage_a_candidates}"
            ))
            
            # Validate Step 6 final count
            final_count = len(final_recommendations)
            expected_final = scenario['validation_checks']['step6_final_count']
            validation_results.append(self._validate_range(
                "step6_final_count", final_count, expected_final,
                f"Final recommendations: {final_count}"
            ))
            
            # Validate evidence alignment
            if final_recommendations:
                evidence_alignment = self.metrics.calculate_evidence_alignment_score(final_recommendations)
                min_alignment = scenario['expected_outcomes']['min_evidence_alignment']
                validation_results.append(ValidationResult(
                    check_name="evidence_alignment",
                    passed=evidence_alignment >= min_alignment,
                    actual_value=evidence_alignment,
                    expected_range={"min": min_alignment},
                    score=min(evidence_alignment / min_alignment, 1.0) if min_alignment > 0 else 1.0,
                    message=f"Evidence alignment: {evidence_alignment:.3f} (min: {min_alignment})"
                ))
                
                # Validate citation coverage
                citation_coverage = self.metrics.calculate_citation_coverage(final_recommendations)
                min_coverage = scenario['validation_checks']['evidence_citation_coverage']['min']
                validation_results.append(ValidationResult(
                    check_name="citation_coverage",
                    passed=citation_coverage >= min_coverage,
                    actual_value=citation_coverage,
                    expected_range={"min": min_coverage},
                    score=min(citation_coverage / min_coverage, 1.0) if min_coverage > 0 else 1.0,
                    message=f"Citation coverage: {citation_coverage:.3f} (min: {min_coverage})"
                ))
                
                # Validate dimensional balance
                balance_metrics = self.metrics.calculate_dimensional_balance(final_recommendations)
                balance_check = scenario['validation_checks']['dimensional_score_balance']
                
                if 'overall_avg' in balance_metrics:
                    overall_avg = balance_metrics['overall_avg']
                    overall_std = balance_metrics['overall_std']
                    
                    avg_passed = overall_avg >= balance_check['min_avg']
                    std_passed = overall_std <= balance_check['max_std']
                    
                    validation_results.append(ValidationResult(
                        check_name="dimensional_balance",
                        passed=avg_passed and std_passed,
                        actual_value={"avg": overall_avg, "std": overall_std},
                        expected_range=balance_check,
                        score=(min(overall_avg / balance_check['min_avg'], 1.0) * 0.7 + 
                              min(balance_check['max_std'] / max(overall_std, 0.01), 1.0) * 0.3),
                        message=f"Dimensional balance: avg={overall_avg:.3f}, std={overall_std:.3f}"
                    ))
                
                # Validate theme alignment
                expected_outcomes = scenario['expected_outcomes']
                theme_metrics = self.metrics.detect_theme_alignment(
                    final_recommendations, 
                    expected_outcomes.get('themes', []),
                    expected_outcomes.get('avoid_themes', [])
                )
                
                theme_match_rate = theme_metrics['theme_match_rate']
                avoid_violation_rate = theme_metrics['avoid_violation_rate']
                
                # Theme alignment score (higher match rate is better, lower violation rate is better)
                theme_score = theme_match_rate * 0.7 + (1.0 - avoid_violation_rate) * 0.3
                
                validation_results.append(ValidationResult(
                    check_name="theme_alignment",
                    passed=theme_match_rate >= 0.3 and avoid_violation_rate <= 0.1,
                    actual_value=theme_metrics,
                    expected_range={"min_match": 0.3, "max_avoid": 0.1},
                    score=theme_score,
                    message=f"Theme alignment: match={theme_match_rate:.3f}, avoid={avoid_violation_rate:.3f}"
                ))
            
            # Validate processing time
            processing_time = pipeline_output.get('total_time', 0)
            max_time = scenario['expected_outcomes']['max_processing_time']
            validation_results.append(ValidationResult(
                check_name="processing_time",
                passed=processing_time <= max_time,
                actual_value=processing_time,
                expected_range={"max": max_time},
                score=max(1.0 - (processing_time / max_time - 1.0), 0.0) if processing_time > max_time else 1.0,
                message=f"Processing time: {processing_time:.2f}s (max: {max_time}s)"
            ))
            
            # Calculate overall score and pass status
            overall_score = statistics.mean([vr.score for vr in validation_results])
            passed = all(vr.passed for vr in validation_results)
            
            return ScenarioResult(
                scenario_id=scenario['scenario_id'],
                test_type=scenario['test_type'],
                passed=passed,
                overall_score=overall_score,
                processing_time=time.time() - start_time,
                validation_results=validation_results,
                pipeline_output=pipeline_output,
                timestamp=datetime.now().isoformat(),
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Validation error for scenario {scenario['scenario_id']}: {e}")
            return ScenarioResult(
                scenario_id=scenario['scenario_id'],
                test_type=scenario['test_type'],
                passed=False,
                overall_score=0.0,
                processing_time=time.time() - start_time,
                validation_results=validation_results,
                pipeline_output=pipeline_output,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
    
    def _validate_range(self, check_name: str, actual_value: Any, 
                       expected_range: Dict[str, Any], message: str) -> ValidationResult:
        """Validate a value against a range specification"""
        passed = True
        score = 1.0
        
        if 'exact' in expected_range:
            passed = actual_value == expected_range['exact']
            score = 1.0 if passed else 0.0
        else:
            if 'min' in expected_range and actual_value < expected_range['min']:
                passed = False
                score = max(actual_value / expected_range['min'], 0.0)
            
            if 'max' in expected_range and actual_value > expected_range['max']:
                passed = False
                score = min(expected_range['max'] / actual_value, score)
        
        return ValidationResult(
            check_name=check_name,
            passed=passed,
            actual_value=actual_value,
            expected_range=expected_range,
            score=score,
            message=message
        )
    
    def generate_test_report(self, results: List[ScenarioResult], 
                           output_path: str = "tests/test_report.json") -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate aggregate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        avg_score = statistics.mean([r.overall_score for r in results]) if results else 0.0
        avg_processing_time = statistics.mean([r.processing_time for r in results]) if results else 0.0
        
        # Group by test type
        by_test_type = {}
        for result in results:
            test_type = result.test_type
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
        
        test_type_summary = {}
        for test_type, type_results in by_test_type.items():
            type_passed = sum(1 for r in type_results if r.passed)
            test_type_summary[test_type] = {
                "total": len(type_results),
                "passed": type_passed,
                "pass_rate": type_passed / len(type_results),
                "avg_score": statistics.mean([r.overall_score for r in type_results])
            }
        
        # Failed tests summary
        failed_tests = [r for r in results if not r.passed]
        failed_summary = []
        for failed in failed_tests:
            failed_checks = [vr for vr in failed.validation_results if not vr.passed]
            failed_summary.append({
                "scenario_id": failed.scenario_id,
                "test_type": failed.test_type,
                "overall_score": failed.overall_score,
                "failed_checks": [{"check": fc.check_name, "message": fc.message} for fc in failed_checks],
                "error_message": failed.error_message
            })
        
        report = {
            "test_run_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": pass_rate,
                "average_score": avg_score,
                "average_processing_time": avg_processing_time
            },
            "test_type_breakdown": test_type_summary,
            "failed_tests": failed_summary,
            "detailed_results": [asdict(r) for r in results]
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test report saved to {output_path}")
        logger.info(f"Test Summary: {passed_tests}/{total_tests} passed ({pass_rate:.1%}), "
                   f"avg score: {avg_score:.3f}")
        
        return report

def main():
    """Example usage of the quality validation system"""
    validator = PipelineValidator()
    
    # Load test scenarios
    scenarios = validator.load_test_scenarios()
    if not scenarios:
        logger.error("No test scenarios loaded")
        return
    
    # Mock pipeline output for demonstration
    mock_pipeline_output = {
        "step5_result": {
            "evidence_count": 12,
            "time": 0.01
        },
        "stage_a_result": {
            "final_candidates": 150,
            "time": 8.14
        },
        "step6_result": {
            "final_recommendations": [
                {
                    "artwork_id": 27307,
                    "llm_score": 0.89,
                    "scores": {
                        "emotional_fit": 0.92,
                        "narrative_fit": 0.87,
                        "subject_fit": 0.90,
                        "palette_fit": 0.85,
                        "style_fit": 0.88,
                        "evidence_alignment": 0.93
                    },
                    "reasoning": "This calming landscape with soft blue tones is ideal for stress relief...",
                    "evidence_used": ["Color psychology workplace study"]
                }
                # ... would have 30 total recommendations
            ] * 30  # Mock 30 recommendations
        },
        "total_time": 9.2
    }
    
    # Run validation on first scenario
    if scenarios:
        first_scenario = scenarios[0]
        result = validator.validate_scenario(first_scenario, mock_pipeline_output)
        
        print(f"\nValidation Result for {result.scenario_id}:")
        print(f"Passed: {result.passed}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        
        print("\nValidation Details:")
        for vr in result.validation_results:
            status = "✅" if vr.passed else "❌"
            print(f"{status} {vr.check_name}: {vr.message} (score: {vr.score:.3f})")

if __name__ == "__main__":
    main()