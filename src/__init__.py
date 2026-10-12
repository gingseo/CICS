"""CICS: reproducible resolution-loss sensitivity experiments."""
import os
# Must precede the first CUDA context, including MMEngine training.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
