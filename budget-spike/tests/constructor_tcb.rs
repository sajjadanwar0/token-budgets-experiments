//! Constructor surface as the trusted computing base (TCB).
//!
//! This file makes the threat model from §III of the paper concrete in
//! the artifact. The affine `Budget` discipline rules out three classes
//! of in-program cap-circumvention at compile time (clone, double-spend,
//! use-after-split — see `tests/compile_fail/`). It does NOT prevent
//! abuse of the `Budget::new` constructor: any code with module-level
//! access to `budget_spike::Budget::new` can mint a budget of arbitrary
//! size.
//!
//! This is a deliberate property of the design, not a bug. The threat
//! model is:
//!
//!     TCB = { Budget::new callers } ∪ { unsafe code in calling crate }
//!
//! What this file demonstrates:
//!
//!  1. `Budget::new(u64::MAX)` succeeds from any caller. (The discipline
//!     does NOT prevent constructor abuse; the constructor is in the TCB.)
//!
//!  2. A "trusted module" pattern that an operator can use to keep the
//!     TCB auditable: wrap `Budget::new` in a single configuration-driven
//!     factory inside a private module, expose only the factory, and
//!     audit the (small) set of files importing the factory.
//!
//!  3. Tools like `cargo geiger` and module-visibility lints make the
//!     trusted set statically discoverable — the TCB is not eliminated,
//!     it is made auditable.
//!
//! These are POSITIVE tests (they compile and pass). The fact that they
//! compile is the point: the discipline does not pretend the constructor
//! is unforgeable.

use budget_spike::{Budget, BudgetError};

// ---------------------------------------------------------------------------
// Test 1: The constructor accepts arbitrary values from any caller.
// ---------------------------------------------------------------------------
//
// This test passes if the Budget::new call type-checks and runs. The fact
// that it does is the artifact-level demonstration of the trust assumption.

#[test]
fn constructor_can_mint_arbitrary_budget() {
    // Any caller can construct a budget of any size. The discipline does
    // not prevent this; preventing it would require a capability-aware
    // language like Pony or Wyvern (see §VI-B of the paper).
    let huge = Budget::new(u64::MAX);
    assert_eq!(huge.available(), u64::MAX);

    let small = Budget::new(1);
    assert_eq!(small.available(), 1);

    let zero = Budget::new(0);
    assert_eq!(zero.available(), 0);

    // What the discipline DOES prevent: aliasing or duplicating an
    // *existing* budget value. The compile-fail tests in
    // tests/compile_fail/ demonstrate the rejected paths. Here we
    // exercise the orthogonal property: that a fresh constructor call
    // always succeeds, no matter the value.
}

// ---------------------------------------------------------------------------
// Test 2: A `Budget::new(u64::MAX)`-budget cannot be aliased even though
//         it can be minted. The two properties are independent.
// ---------------------------------------------------------------------------

#[test]
fn maxed_budget_still_obeys_affine_discipline() {
    let b = Budget::new(u64::MAX);

    // The affine discipline applies to b regardless of how it was
    // constructed. Splitting consumes self:
    let (parent, child) = b.split(1_000_000).expect("split should succeed");
    assert_eq!(parent.available(), u64::MAX - 1_000_000);
    assert_eq!(child.available(), 1_000_000);

    // The original `b` is now consumed; attempting to use it would be
    // an E0382 compile error. (See tests/compile_fail/spend_after_split.rs
    // for the compile-fail demonstration.)
    //
    // Compile-time discipline is orthogonal to constructor authority:
    // a maliciously-minted Budget is still subject to the no-aliasing
    // rule, which means the attacker cannot fan out their illegitimate
    // budget into many copies — they can only mint once per call site.

    // Continue with the descendants:
    let (parent, _) = parent.spend(500, || ()).expect("spend should succeed");
    let _ = child.consume();
    let _ = parent.consume();
}

// ---------------------------------------------------------------------------
// Test 3: A "trusted-module" pattern that operators can use to confine
//         the constructor surface. Recommended deployment idiom.
// ---------------------------------------------------------------------------
//
// The deployment-side TCB is the set of files invoking `Budget::new`.
// Keeping that set small is a project-level discipline, not a type-system
// guarantee. The pattern below shows the recommended idiom: wrap
// `Budget::new` inside one private module, expose only a configuration-
// driven factory, and rely on Rust's module visibility to make the TCB
// statically discoverable.

mod trusted_factory {
    //! The single trusted module in this hypothetical deployment.
    //!
    //! In a real codebase, this module would live in a top-level
    //! `crate::trusted_budget` and would be the ONLY file in the crate
    //! that imports `budget_spike::Budget::new`. A `cargo geiger` audit
    //! plus `grep -r "Budget::new"` over the crate gives the operator
    //! a concrete TCB list.

    use budget_spike::Budget;

    /// Configuration for the trusted budget factory.
    pub struct BudgetPolicy {
        /// Hard cap on any single Budget the factory will mint.
        pub max_budget_uc: u64,
        /// Per-request default budget if no override is supplied.
        pub default_budget_uc: u64,
    }

    impl BudgetPolicy {
        /// Mint a budget. Returns Err if the requested amount exceeds the
        /// policy cap; the factory itself enforces the policy.
        pub fn mint(&self, amount_uc: u64) -> Result<Budget, &'static str> {
            if amount_uc > self.max_budget_uc {
                return Err("requested amount exceeds policy cap");
            }
            Ok(Budget::new(amount_uc))
        }

        /// Mint the default budget. Always succeeds, since the default
        /// is always within the policy cap by construction.
        pub fn mint_default(&self) -> Budget {
            Budget::new(self.default_budget_uc.min(self.max_budget_uc))
        }
    }
}

#[test]
fn trusted_factory_pattern_works() {
    use trusted_factory::BudgetPolicy;

    let policy = BudgetPolicy {
        max_budget_uc: 10_000_000, // $10.00 cap per Budget
        default_budget_uc: 1_000_000, // $1.00 default
    };

    // Within-policy mint succeeds.
    let b = policy.mint(5_000_000).expect("within policy");
    assert_eq!(b.available(), 5_000_000);
    let _ = b.consume();

    // Above-policy mint is rejected by the factory itself, not by the
    // type system. The factory is the project-level TCB; the type system
    // ensures the resulting Budget cannot be aliased once minted.
    let err = policy.mint(20_000_000).expect_err("over policy");
    assert_eq!(err, "requested amount exceeds policy cap");

    // Default mint is always within policy.
    let b = policy.mint_default();
    assert_eq!(b.available(), 1_000_000);
    let _ = b.consume();
}

// ---------------------------------------------------------------------------
// Test 4: An adversarial caller can still bypass the trusted factory if
//         they have access to `Budget::new` directly. This test
//         documents that the type system cannot prevent this.
// ---------------------------------------------------------------------------

#[test]
fn adversarial_caller_can_bypass_trusted_factory() {
    // Suppose the trusted factory caps individual budgets at $10. A
    // caller with direct access to Budget::new can simply ignore the
    // factory and mint whatever they want. The discipline does NOT
    // prevent this; that is the whole point of stating that the
    // constructor is in the TCB.
    //
    // The defense is project-level: lint for direct `Budget::new`
    // calls outside the trusted module, fail CI on violations.

    let factory = trusted_factory::BudgetPolicy {
        max_budget_uc: 10_000_000,
        default_budget_uc: 1_000_000,
    };
    assert!(factory.mint(20_000_000).is_err()); // factory says no

    let bypass = Budget::new(20_000_000); // type system says yes
    assert_eq!(bypass.available(), 20_000_000);

    // But once minted, even the adversarial budget is subject to the
    // affine discipline. The attacker can only spend it; they cannot
    // duplicate it into many copies.
    let result = bypass.spend(20_000_001, || ());
    assert!(matches!(result, Err(BudgetError::Insufficient { .. })));
}

// ---------------------------------------------------------------------------
// Test 5: The constructor's trust assumption is orthogonal to the
//         conservation invariant maintained by split/merge.
// ---------------------------------------------------------------------------

#[test]
fn conservation_holds_regardless_of_constructor_input() {
    // The split-merge round-trip preserves the original quota, no matter
    // how the original was constructed. This is the conservation
    // invariant verified by Lemma 1's TLA+/Coq/Dafny proofs.

    let b = Budget::new(1_000_000);
    let (parent, child) = b.split(300_000).expect("split");
    let merged = parent.merge(child);
    assert_eq!(merged.available(), 1_000_000); // conservation holds

    // Even with an adversarially-large initial budget, conservation
    // holds across the split/merge boundary:
    let b = Budget::new(u64::MAX / 2); // within Assumption A2
    let (parent, child) = b.split(1_000).expect("split");
    let merged = parent.merge(child);
    assert_eq!(merged.available(), u64::MAX / 2);
    let _ = merged.consume();
}