#!/usr/bin/env python3
"""
Test runner for warp bubble QFT implementation.
"""

import sys
import os
import subprocess
import argparse
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_file(test_file, verbose=True):
    """Run a single test file and return success status."""
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {test_file}")
            print(f"{'='*60}")
        
        start_time = time.time()
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            if verbose:
                print(f"✅ {test_file} PASSED ({elapsed_time:.2f}s)")
                if result.stdout and verbose > 1:
                    print(result.stdout)
            return True
        else:
            if verbose:
                print(f"❌ {test_file} FAILED ({elapsed_time:.2f}s)")
                if result.stderr:
                    print("STDERR:", result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        if verbose:
            print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Run Warp Bubble QFT Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run only core tests")
    parser.add_argument("--verbose", "-v", action="count", default=1, help="Increase verbosity")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--demo", action="store_true", help="Run demo files only")
    
    args = parser.parse_args()
    
    print("🧪 Running Warp Bubble QFT Test Suite")
    print("=" * 60)
    
    # Test files to run
    test_files = [
        "tests/test_recent_discoveries.py",
        "tests/test_field_algebra.py", 
        "tests/test_field_commutators.py",
        "tests/test_negative_energy.py",
        "tests/test_negative_energy_bounds.py"
    ]
    
    # Demo files to test
    demo_files = [
        "examples/demo_warp_bubble_sim.py",
        "examples/demo_3d_negative_energy.py",
        "debug_energy.py"
    ]
    
    # If specific file specified, only run that
    if args.file:
        if os.path.exists(args.file):
            success = run_test_file(args.file, args.verbose)
            return 0 if success else 1
        else:
            print(f"⚠️  Test file not found: {args.file}")
            return 1
    
    # If demo flag, only run demos
    if args.demo:
        files_to_run = demo_files
    else:
        files_to_run = test_files
        if not args.quick:
            files_to_run += demo_files
    
    passed = 0
    failed = 0
    
    # Run tests
    if not args.demo:
        print("\n🔬 Running Core Tests")
    
    for test_file in files_to_run:
        if os.path.exists(test_file):
            if run_test_file(test_file, args.verbose):
                passed += 1
            else:
                failed += 1
                if test_file in demo_files:
                    print(f"   (Demo failure - not critical)")
        else:
            print(f"⚠️  Test file not found: {test_file}")
            if test_file not in demo_files:
                failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "No tests run")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


def run_subset(subset_name, verbose=True):
    """Run a predefined subset of tests."""
    if subset_name == "field":
        test_files = [
            "tests/test_field_algebra.py",
            "tests/test_field_commutators.py"
        ]
    elif subset_name == "energy":
        test_files = [
            "tests/test_negative_energy.py",
            "tests/test_negative_energy_bounds.py"
        ]
    elif subset_name == "discoveries":
        test_files = [
            "tests/test_recent_discoveries.py"
        ]
    elif subset_name == "demos":
        test_files = [
            "examples/demo_warp_bubble_sim.py",
            "examples/demo_3d_negative_energy.py"
        ]
    else:
        print(f"Unknown test subset: {subset_name}")
        return 1
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test_file(test_file, verbose):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  Test file not found: {test_file}")
            failed += 1
    
    print(f"\n{subset_name.upper()} tests: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
