"""EvaluatorFactory — creates Evaluator instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError

if TYPE_CHECKING:
    from src.core.settings import EvaluationSettings


_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}

# _builtins_registered 标记内置评估器是否已惰性导入注册
_builtins_registered = False


def register_evaluator(provider: str):
    """Decorator to register an evaluator implementation."""
    def decorator(cls: type[BaseEvaluator]):
        _EVALUATOR_REGISTRY[provider.lower()] = cls
        return cls
    return decorator


def _ensure_builtins_registered() -> None:
    """Lazily import built-in evaluator modules to trigger their registration.

    Registration relies on the @register_evaluator decorator running at import
    time. Production code paths may not import the implementation modules
    directly, so import them here on demand (idempotent).
    """
    global _builtins_registered
    if _builtins_registered:
        return
    # Import for side effects (decorator registration). Failures are tolerated
    # so a missing optional backend never blocks the others.
    try:
        import src.libs.evaluator.custom_evaluator  # noqa: F401
    except ImportError:
        pass
    try:
        import src.observability.evaluation.ragas_evaluator  # noqa: F401
    except ImportError:
        pass
    _builtins_registered = True


class EvaluatorFactory:
    """Factory for creating Evaluator instances based on settings."""

    @staticmethod
    def create(backend: str) -> BaseEvaluator:
        """Create an Evaluator instance for the specified backend.

        Args:
            backend: Evaluator backend name (e.g., 'custom', 'ragas').

        Returns:
            An instance of BaseEvaluator.

        Raises:
            EvaluatorError: If the backend is unknown.
        """
        _ensure_builtins_registered()
        backend_lower = backend.lower()

        if backend_lower not in _EVALUATOR_REGISTRY:
            available = ", ".join(sorted(_EVALUATOR_REGISTRY.keys())) or "(none registered)"
            raise EvaluatorError(
                f"Unknown evaluator backend: '{backend}'. Available: {available}",
                provider=backend,
            )

        cls = _EVALUATOR_REGISTRY[backend_lower]
        return cls()

    @staticmethod
    def create_composite(backends: list[str]) -> BaseEvaluator:
        """Create a CompositeEvaluator wrapping the given backends.

        Args:
            backends: Ordered list of evaluator backend names.

        Returns:
            A CompositeEvaluator combining the requested backends.

        Raises:
            EvaluatorError: If any backend is unknown or the list is empty.
        """
        if not backends:
            raise EvaluatorError("No evaluator backends configured")

        from src.observability.evaluation.composite_evaluator import CompositeEvaluator

        evaluators = [EvaluatorFactory.create(name) for name in backends]
        return CompositeEvaluator(evaluators)

    @staticmethod
    def available_backends() -> list[str]:
        """Return list of registered backend names."""
        _ensure_builtins_registered()
        return sorted(_EVALUATOR_REGISTRY.keys())
