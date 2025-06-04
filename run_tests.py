#!/usr/bin/env python3
"""
Test runner for warp bubble QFT implementation.
"""

import sys
import os
import subprocess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_file(test_file):
    """Run a single test file and return success status."""
    try:
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")
        
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_file} FAILED")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
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
        "debug_energy.py"
    ]
    
    passed = 0
    failed = 0
    
    # Run core tests
    print("\n🔬 Running Core Tests")
    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  Test file not found: {test_file}")
            failed += 1
    
    # Run demo tests (non-critical)
    print("\n🎮 Running Demo Tests")
    for demo_file in demo_files:
        if os.path.exists(demo_file):
            if run_test_file(demo_file):
                passed += 1
            else:
                failed += 1
                print(f"   (Demo failure - not critical)")
        else:
            print(f"⚠️  Demo file not found: {demo_file}")
    
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

if __name__ == "__main__":
    sys.exit(main())
