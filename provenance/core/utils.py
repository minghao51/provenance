"""Shared utilities for provenance core."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from functools import lru_cache
from typing import Any


class weak_lru_cache:  # noqa: N801
    def __init__(self, maxsize: int = 128, typed: bool = False):
        self.maxsize = maxsize
        self.typed = typed

    def __get__(self, instance: Any, objtype: type | None = None) -> Callable:
        if instance is None:
            return self  # type: ignore[return-value]
        if instance not in self.caches:
            @lru_cache(maxsize=self.maxsize, typed=self.typed)
            def _bound_cache(*args: Any, **kwargs: Any) -> Any:
                return self.func(instance, *args, **kwargs)

            self.caches[instance] = _bound_cache
        return self.caches[instance]

    def __call__(self, func: Callable) -> weak_lru_cache:
        self.func = func
        self.caches: weakref.WeakKeyDictionary[Any, Callable] = (
            weakref.WeakKeyDictionary()
        )
        return self
