---
id: 0006
title: TrackState converted from @dataclass to a plain class
status: accepted
date: 2026-04-18
---

## Status

Accepted. Implemented in the ADR-0005 EKF migration commit (`e85ec90`).

## Context

ADR-0005 specifies that `TrackState.uncertainty_km` becomes a read/write property backed by the covariance matrix, with a setter that projects a scalar km to an isotropic 2×2 position block. It also specifies that the constructor kwarg `TrackState(uncertainty_km=N)` continues to work for backward compatibility with every existing test and production call site.

`@dataclass` does not cleanly support a property-with-setter on the same name as a constructor kwarg:

- If `uncertainty_km` is declared as a regular dataclass field, it cannot also be a `@property` — the class attribute would be the property descriptor, not the default value, and the generated `__init__` would fail.
- Declaring it as `InitVar[Optional[float]]` makes `uncertainty_km` a pseudo-field whose value is passed to `__post_init__` instead of being stored as an attribute. This allows the property to coexist under the same name, but requires routing the init value through a `__post_init__` method and calling the property setter explicitly via `type(self).uncertainty_km.fset(self, value)` — a non-obvious pattern that obscures the EKF initialization flow for readers.
- Order-of-definition pitfalls with `@dataclass` decorators and class-level property declarations are easy to get wrong in subtle ways.

The EKF initialization logic is the load-bearing piece of Week 1's belief-state math. Hiding its branching (cov-supplied vs uncertainty_km-supplied vs default) inside `__post_init__` gymnastics would make the class harder to audit and easier to break.

## Decision

`TrackState` is a plain class with an explicit `__init__` that handles the three construction paths — `cov=`, `uncertainty_km=`, or neither — directly. `__repr__` is preserved. Auto-generated `__eq__` from the prior `@dataclass` decorator is not preserved; equality is identity-based (the default `object.__eq__`).

## Consequences

- EKF initialization flow is a straightforward if/elif/else in the constructor, auditable at the call site with no `__post_init__` indirection.
- `uncertainty_km` as a property-with-setter coexists cleanly with `uncertainty_km` as a constructor kwarg — both names point at the same shim projection logic.
- Equality comparison between `TrackState` instances is now identity-based (`a == b` iff `a is b`). No current code in the repo relies on value-based `TrackState` equality; the full test suite (2067 tests) passes without change. If future code needs structural equality, it must implement `__eq__` explicitly — and that code should have an ADR of its own for the semantic ambiguity (e.g., do two tracks with identical cov but different `last_collection_time` count as equal?).
- All other public surface — methods, properties, field names, constructor kwargs — is preserved. External callers see no change other than the equality behavior noted above.
- Auto-`__hash__` is now `object.__hash__` (identity-based), which is consistent with identity-based `__eq__`. Dict/set membership by identity is preserved; by-value membership was never a usage pattern here.
- Serialization and debugging affordances: `__repr__` is explicit and reports `uncertainty_km`, `confidence`, `consecutive_failures`, and `is_dark` — the same signals the old auto-generated `__repr__` exposed.
