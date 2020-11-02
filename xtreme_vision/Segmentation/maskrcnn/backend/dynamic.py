_BACKEND = "tensorflow"

if _BACKEND == "tensorflow":
    from .tensorflow_backend import *
else:
    raise ValueError("Unknown backend: " + str(_BACKEND))
