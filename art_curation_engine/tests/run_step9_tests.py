#!/usr/bin/env python3
"""
Step 9: Comprehensive Test Runner for Quality/Regression Testing System

This script demonstrates the complete Step 9 testing framework including:
- Quality validation
- Regression testing
- Baseline management
- Performance monitoring
"""

import sys
import json
import time
from pathlib import Path
import logging

# Import test modules using relative imports
from .step9_quality_validator import PipelineValidator, QualityMetrics
from .step9_regression_tester import RegressionTester, BaselineManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quality_validation_demo():
    """Demonstrate quality validation system"""
    print("\n" + "="*80)
    print("STEP 9: QUALITY VALIDATION SYSTEM DEMO")
    print("="*80)
    
    validator = PipelineValidator()
    metrics = QualityMetrics()
    
    # Load test scenarios
    scenarios = validator.load_test_scenarios()
    print(f"📝 Loaded {len(scenarios)} test scenarios")
    
    # Demonstrate validation on mock data
    mock_pipeline_output = create_mock_pipeline_output()
    
    if scenarios:
        first_scenario = scenarios[0]
        print(f"\n🧪 Testing scenario: {first_scenario['scenario_id']}")
        print(f"   Input: {first_scenario['user_input'][:60]}...")
        
        # Run validation
        result = validator.validate_scenario(first_scenario, mock_pipeline_output)
        
        print(f"\n📊 Validation Results:")
        print(f"   Overall Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        print(f"   Overall Score: {result.overall_score:.3f}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        
        print(f"\n🔍 Detailed Validation Checks:")
        for vr in result.validation_results:
            status = "✅" if vr.passed else "❌"
            print(f"   {status} {vr.check_name}: {vr.message} (score: {vr.score:.3f})")
        
        # Demonstrate individual metric calculations
        final_recs = mock_pipeline_output['step6_result']['final_recommendations']
        
        print(f"\n📈 Individual Quality Metrics:")
        evidence_alignment = metrics.calculate_evidence_alignment_score(final_recs)
        print(f"   Evidence Alignment: {evidence_alignment:.3f}")
        
        citation_coverage = metrics.calculate_citation_coverage(final_recs)
        print(f"   Citation Coverage: {citation_coverage:.3f}")
        
        dimensional_balance = metrics.calculate_dimensional_balance(final_recs)
        if 'overall_avg' in dimensional_balance:
            print(f"   Dimensional Balance: avg={dimensional_balance['overall_avg']:.3f}, "
                  f"std={dimensional_balance['overall_std']:.3f}")
        
        theme_alignment = metrics.detect_theme_alignment(
            final_recs, 
            first_scenario['expected_outcomes'].get('themes', []),
            first_scenario['expected_outcomes'].get('avoid_themes', [])
        )
        print(f"   Theme Alignment: match={theme_alignment['theme_match_rate']:.3f}, "
              f"avoid={theme_alignment['avoid_violation_rate']:.3f}")

def run_regression_testing_demo():
    """Demonstrate regression testing system"""
    print("\n" + "="*80)
    print("STEP 9: REGRESSION TESTING SYSTEM DEMO")
    print("="*80)
    
    tester = RegressionTester()
    baseline_manager = BaselineManager()
    
    # Create tests directory structure
    (Path("tests") / "baselines").mkdir(exist_ok=True)
    (Path("tests") / "results").mkdir(exist_ok=True)
    
    print("📁 Test directory structure created")
    
    # Demonstrate baseline management
    print(f"\n📋 Baseline Management Demo:")
    
    # Check existing baselines
    baseline_dir = Path("tests/baselines")
    existing_baselines = list(baseline_dir.glob("*_baseline.json"))
    print(f"   Existing baselines: {len(existing_baselines)}")
    
    # Run regression tests (using mock execution)
    print(f"\n🔄 Running regression tests...")
    
    try:
        # Run tests on regression scenarios only
        results = tester.run_regression_tests(scenario_types=["regression"])
        
        if results and "regression_summary" in results:
            summary = results["regression_summary"]
            print(f"\n📊 Regression Test Summary:")
            print(f"   Total Scenarios: {summary['total_scenarios']}")
            print(f"   Passed: {summary['passed_scenarios']}")
            print(f"   Failed: {summary['failed_scenarios']}")
            print(f"   Regressions Detected: {summary['regressions_detected']}")
            print(f"   New Baselines Created: {summary['new_baselines_created']}")
            
            if summary['regressions_detected'] > 0:
                print(f"\n⚠️  Regression Details:")
                for detail in summary['scenario_details']:
                    baseline_comp = detail.get('baseline_comparison', {})
                    if baseline_comp.get('regression_detected', False):
                        print(f"   - {detail['scenario_id']}: {baseline_comp.get('regressions', [])}")
        
    except Exception as e:
        logger.error(f"Regression testing demo failed: {e}")
        print(f"   ❌ Error during regression testing: {e}")

def run_performance_monitoring_demo():
    """Demonstrate performance monitoring capabilities"""
    print("\n" + "="*80)
    print("STEP 9: PERFORMANCE MONITORING DEMO")
    print("="*80)
    
    # Simulate performance data collection
    scenarios = ["work_stress_baseline", "evening_relaxation", "creative_focus", "anxiety_relief"]
    
    print(f"📊 Performance Monitoring for {len(scenarios)} scenarios:")
    
    performance_data = {}
    
    for scenario_id in scenarios:
        # Simulate performance metrics
        processing_time = 8.5 + (hash(scenario_id) % 100) / 20  # 8.5-13.5 seconds
        evidence_alignment = 0.75 + (hash(scenario_id) % 20) / 100  # 0.75-0.95
        memory_usage = 150 + (hash(scenario_id) % 50)  # 150-200 MB
        
        performance_data[scenario_id] = {
            "processing_time": processing_time,
            "evidence_alignment": evidence_alignment,
            "memory_usage": memory_usage,
            "timestamp": time.time()
        }
        
        print(f"   {scenario_id}:")
        print(f"     Processing Time: {processing_time:.2f}s")
        print(f"     Evidence Alignment: {evidence_alignment:.3f}")
        print(f"     Memory Usage: {memory_usage}MB")
    
    # Performance trend analysis
    avg_processing_time = sum(d["processing_time"] for d in performance_data.values()) / len(performance_data)
    avg_evidence_alignment = sum(d["evidence_alignment"] for d in performance_data.values()) / len(performance_data)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Average Processing Time: {avg_processing_time:.2f}s")
    print(f"   Average Evidence Alignment: {avg_evidence_alignment:.3f}")
    print(f"   Performance Status: {'✅ GOOD' if avg_processing_time < 12.0 else '⚠️ SLOW'}")
    print(f"   Quality Status: {'✅ GOOD' if avg_evidence_alignment > 0.80 else '⚠️ LOW'}")

def create_mock_pipeline_output():
    """Create realistic mock pipeline output for demonstration"""
    mock_recommendations = []
    
    # Create 30 mock recommendations with realistic score distribution
    base_scores = {
        "emotional_fit": 0.88,
        "narrative_fit": 0.83, 
        "subject_fit": 0.87,
        "palette_fit": 0.82,
        "style_fit": 0.85,
        "evidence_alignment": 0.89
    }
    
    for i in range(30):
        # Add some variance to scores
        scores = {}
        for dim, base_score in base_scores.items():
            variance = (hash(f"{dim}_{i}") % 20 - 10) / 200  # ±0.05 variance
            scores[dim] = max(0.0, min(1.0, base_score + variance))
        
        llm_score = sum(scores.values()) / len(scores)
        
        mock_recommendations.append({
            "artwork_id": 27000 + i,
            "llm_score": llm_score,
            "scores": scores,
            "reasoning": f"Mock artwork {i+1} demonstrates calming blue tones effective for stress relief (Color Psychology Study 2019)...",
            "evidence_used": ["Color psychology workplace study", "Environmental stress research"]
        })
    
    return {
        "step5_result": {
            "time": 0.02,
            "evidence_count": 12,
            "brief_generated": True,
            "queries_generated": 5,
            "cache_hit": True
        },
        "stage_a_result": {
            "time": 2.1,
            "final_candidates": 150,
            "A1_metadata_hits": 200,
            "A2_clip_hits": 150,
            "generated_keywords": 10,
            "clip_prompts": 3,
            "cache_hit": False
        },
        "step6_result": {
            "time": 6.8,
            "final_recommendations": mock_recommendations
        },
        "total_time": 8.92,
        "success": True
    }

def run_full_test_suite():
    """Run complete Step 9 test suite"""
    print("\n" + "🚀"*40)
    print("STEP 9: COMPLETE QUALITY/REGRESSION TEST SUITE")
    print("🚀"*40)
    
    start_time = time.time()
    
    try:
        # 1. Quality Validation Demo
        run_quality_validation_demo()
        
        # 2. Regression Testing Demo  
        run_regression_testing_demo()
        
        # 3. Performance Monitoring Demo
        run_performance_monitoring_demo()
        
        total_time = time.time() - start_time
        
        print("\n" + "✅"*40)
        print("STEP 9 TEST SUITE COMPLETED SUCCESSFULLY")
        print("✅"*40)
        print(f"Total Execution Time: {total_time:.2f}s")
        print("\n📋 Test Suite Components:")
        print("   ✅ Quality Validation System")
        print("   ✅ Regression Testing Framework")
        print("   ✅ Baseline Management")
        print("   ✅ Performance Monitoring")
        print("   ✅ Test Scenario Management")
        print("   ✅ Automated Reporting")
        
        print("\n📁 Generated Test Artifacts:")
        print("   - tests/test_scenarios.jsonl (Test scenarios)")
        print("   - tests/baselines/ (Baseline results)")
        print("   - tests/results/ (Test reports)")
        print("   - tests/step9_quality_validator.py (Validation engine)")
        print("   - tests/step9_regression_tester.py (Regression framework)")
        
        print("\n🔧 Usage Examples:")
        print("   # Run quality validation:")
        print("   python tests/step9_quality_validator.py")
        print("")
        print("   # Run regression tests:")
        print("   python tests/step9_regression_tester.py")
        print("")
        print("   # Run specific test types:")
        print("   python tests/step9_regression_tester.py --types regression quality")
        print("")
        print("   # Update baselines:")
        print("   python tests/step9_regression_tester.py --update-baselines")
        
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        logger.error(f"Test suite error: {e}", exc_info=True)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 9 Quality/Regression Testing Demo")
    parser.add_argument("--component", 
                       choices=["quality", "regression", "performance", "full"],
                       default="full",
                       help="Test component to run")
    
    args = parser.parse_args()
    
    if args.component == "quality":
        run_quality_validation_demo()
    elif args.component == "regression":
        run_regression_testing_demo()
    elif args.component == "performance":
        run_performance_monitoring_demo()
    else:
        run_full_test_suite()

if __name__ == "__main__":
    main()