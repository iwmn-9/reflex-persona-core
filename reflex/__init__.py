"""Finite, batched personality-conditioned game decisions. No runtime LLM."""
from .core import NEEDS, TRAITS, VALUES, VERSION, Policy, compile_batch
from .runtime import Population

__all__ = ["NEEDS", "TRAITS", "VALUES", "VERSION", "Policy", "Population", "compile_batch"]
