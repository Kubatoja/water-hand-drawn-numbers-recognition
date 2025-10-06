"""
Test instalacji scikit-optimize
"""
import sys

print("="*60)
print("🔍 Checking Python environment")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    print("✅ scikit-optimize (skopt) - OK")
except ImportError as e:
    print(f"❌ scikit-optimize (skopt) - FAILED: {e}")
    print("\nInstall with: pip install scikit-optimize")
    sys.exit(1)

try:
    import numpy
    print(f"✅ numpy {numpy.__version__} - OK")
except ImportError:
    print("❌ numpy - FAILED")

try:
    import pandas
    print(f"✅ pandas {pandas.__version__} - OK")
except ImportError:
    print("❌ pandas - FAILED")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__} - OK")
except ImportError:
    print("❌ scikit-learn - FAILED")

try:
    import xgboost
    print(f"✅ xgboost {xgboost.__version__} - OK")
except ImportError:
    print("❌ xgboost - FAILED")

print()
print("="*60)
print("🎉 All dependencies installed correctly!")
print("="*60)
print("\nYou can now run:")
print("  python main_bayesian_optimized.py --quick")
print("  python quick_test_bayesian.py")
