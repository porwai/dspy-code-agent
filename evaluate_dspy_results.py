#!/usr/bin/env python3
"""Evaluate DSPy results and save evaluation report."""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def evaluate_dspy_results(preds_path: str, output_dir: str = None):
    """Evaluate DSPy results and save evaluation report."""
    
    preds_file = Path(preds_path)
    if not preds_file.exists():
        print(f"Error: Predictions file not found at {preds_file}")
        return
    
    # Use the same directory as preds.json if no output dir specified
    if output_dir is None:
        output_dir = preds_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating predictions from: {preds_file}")
    print(f"Saving results to: {output_dir}")
    
    # Load predictions
    with open(preds_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Found {len(predictions)} predictions")
    
    # Create evaluation report
    evaluation_report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_predictions": len(predictions),
        "predictions": {},
        "summary": {}
    }
    
    # Analyze each prediction
    successful_predictions = 0
    error_predictions = 0
    rate_limit_errors = 0
    other_errors = 0
    
    for instance_id, pred_data in predictions.items():
        model_patch = pred_data.get("model_patch", "")
        
        # Categorize the prediction
        if "RateLimitError" in model_patch or "rate limit" in model_patch.lower():
            status = "rate_limit_error"
            rate_limit_errors += 1
        elif "Error" in model_patch or "Exception" in model_patch:
            status = "error"
            error_predictions += 1
        elif model_patch.startswith("diff --git") or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in model_patch:
            status = "success"
            successful_predictions += 1
        else:
            status = "other"
            other_errors += 1
        
        evaluation_report["predictions"][instance_id] = {
            "status": status,
            "model_name": pred_data.get("model_name_or_path", "unknown"),
            "has_patch": model_patch.startswith("diff --git"),
            "patch_length": len(model_patch),
            "is_error": status in ["error", "rate_limit_error"]
        }
    
    # Create summary
    evaluation_report["summary"] = {
        "successful_predictions": successful_predictions,
        "error_predictions": error_predictions,
        "rate_limit_errors": rate_limit_errors,
        "other_errors": other_errors,
        "success_rate": successful_predictions / len(predictions) if predictions else 0,
        "error_rate": (error_predictions + rate_limit_errors + other_errors) / len(predictions) if predictions else 0
    }
    
    # Save evaluation report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Create a human-readable summary
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"DSPy Agent Evaluation Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total Predictions: {len(predictions)}\n")
        f.write(f"Successful Predictions: {successful_predictions}\n")
        f.write(f"Error Predictions: {error_predictions}\n")
        f.write(f"Rate Limit Errors: {rate_limit_errors}\n")
        f.write(f"Other Errors: {other_errors}\n")
        f.write(f"Success Rate: {evaluation_report['summary']['success_rate']:.2%}\n")
        f.write(f"Error Rate: {evaluation_report['summary']['error_rate']:.2%}\n\n")
        
        f.write("Instance Details:\n")
        f.write("-" * 30 + "\n")
        for instance_id, details in evaluation_report["predictions"].items():
            f.write(f"{instance_id}: {details['status']}\n")
            if details['is_error']:
                f.write(f"  Model: {details['model_name']}\n")
                f.write(f"  Has Patch: {details['has_patch']}\n")
    
    print(f"\nEvaluation completed!")
    print(f"Report saved to: {report_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nResults Summary:")
    print(f"  Total Predictions: {len(predictions)}")
    print(f"  Successful: {successful_predictions}")
    print(f"  Errors: {error_predictions}")
    print(f"  Rate Limit Errors: {rate_limit_errors}")
    print(f"  Other Errors: {other_errors}")
    print(f"  Success Rate: {evaluation_report['summary']['success_rate']:.2%}")
    
    return evaluation_report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DSPy results")
    parser.add_argument("--preds", required=True, help="Path to preds.json file")
    parser.add_argument("--output", help="Output directory (default: same as preds.json)")
    
    args = parser.parse_args()
    
    evaluate_dspy_results(args.preds, args.output)

if __name__ == "__main__":
    main()
