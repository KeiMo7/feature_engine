"""JSON safe encoding / decoding of transformer parameters and dtypes.

Only structure and parameters are ever serialised: no rows of data are read or
written by this module.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# dtypes that json.dumps cannot handle natively
_NON_JSON_TYPES = (np.ndarray, np.generic, pd.Series, pd.Index, set, tuple)


class _NotSerializable:
    """Sentinel value stored when a parameter cannot be ported to JSON."""

    def __init__(self, reason):
        self.reason = reason


def encode_value(value: Any) -> Any:
    """Convert a fit parameter into a structure made only of JSON primitives."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return encode_value(value.item())

    if isinstance(value, np.ndarray):
        return {"__type__": "numpy.ndarray", "items": [encode_value(v) for v in value]}

    if isinstance(value, (pd.Series, pd.Index)):
        return {
            "__type__": type(value).__name__,
            "items": [encode_value(v) for v in value.tolist()],
        }

    if isinstance(value, (set, tuple)):
        return {
            "__type__": type(value).__name__,
            "items": [encode_value(v) for v in value],
        }

    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [[encode_value(k), encode_value(v)] for k, v in value.items()],
        }

    if isinstance(value, list):
        return [encode_value(v) for v in value]

    if isinstance(value, BaseEstimator):
        return {
            "__type__": "estimator",
            "class_path": _class_path(value),
            "params": {
                k: encode_value(v) for k, v in value.get_params(deep=True).items()
            },
        }

    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module is None or module == "__main__" or "<" in (qualname or ""):
            # lambdas, local functions and partials cannot be imported back.
            return {
                "__type__": "unserializable",
                "kind": type(value).__name__,
                "reason": (
                    "callable is a lambda, a local function or otherwise not "
                    "importable; replace it with a top-level function or a string "
                    "alias recognised by the transformer"
                ),
            }
        return {"__type__": "callable", "module": module, "qualname": qualname}

    return {
        "__type__": "unserializable",
        "kind": type(value).__name__,
        "reason": "value of this type has no portable JSON representation",
    }


def decode_value(value: Any) -> Any:
    """Inverse of :func:`encode_value`."""

    if isinstance(value, list):
        return [decode_value(v) for v in value]

    if not isinstance(value, dict) or "__type__" not in value:
        return value

    tag = value["__type__"]

    if tag == "dict":
        return {decode_value(k): decode_value(v) for k, v in value["items"]}

    if tag == "numpy.ndarray":
        return np.array([decode_value(v) for v in value["items"]])

    if tag == "Series":
        return pd.Series([decode_value(v) for v in value["items"]])

    if tag == "Index":
        return pd.Index([decode_value(v) for v in value["items"]])

    if tag in ("set", "tuple"):
        items = [decode_value(v) for v in value["items"]]
        return set(items) if tag == "set" else tuple(items)

    if tag == "callable":
        return _import_path(value["module"], value["qualname"])

    if tag == "estimator":
        estimator_class = _import_class_path(value["class_path"])
        params = {k: decode_value(v) for k, v in value["params"].items()}
        return estimator_class(**params)

    if tag == "unserializable":
        raise ValueError(
            "Snapshot contains a parameter that could not be serialised when it "
            f"was created ({value.get('reason', 'unknown reason')}). Recreate the "
            "snapshot with a portable parameter value before rolling back."
        )

    raise ValueError(f"Unknown encoded value type '{tag}' found in snapshot.")


def encode_dtype(dtype) -> str:
    """String representation of a pandas / numpy dtype."""

    return str(dtype)


def _class_path(estimator) -> str:
    return f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}"


def _import_class_path(class_path: str):
    module_path, _, qualname = class_path.rpartition(".")
    return _import_path(module_path, qualname)


def _import_path(module_path: str, qualname: str):
    import importlib

    module = importlib.import_module(module_path)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
