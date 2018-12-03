import os
from os.path import isfile, join

preprocessingPath = os.path.dirname(os.path.realpath(__file__))
__all__ = [f[:-3] for f in os.listdir(preprocessingPath) if isfile(join(preprocessingPath, f))]
