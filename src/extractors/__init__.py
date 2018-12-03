import os
from os.path import isfile, join

extractorsPath = os.path.dirname(os.path.realpath(__file__))
__all__ = [f[:-3] for f in os.listdir(extractorsPath) if isfile(join(extractorsPath, f))]
