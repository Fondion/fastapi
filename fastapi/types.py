import types
from enum import Enum
from typing import Any, Callable, Dict, Set, Type, TypeVar, Union

# ModelNameMap is only used with v1 models and due circular import has to be tried out here
try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

DecoratedCallable = TypeVar("DecoratedCallable", bound=Callable[..., Any])
UnionType = getattr(types, "UnionType", Union)
ModelNameMap = Dict[Union[Type[BaseModel], Type[Enum]], str]
IncEx = Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any]]
