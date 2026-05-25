use budget_spike::{Budget, BudgetError};

#[test]
fn constructor_can_mint_arbitrary_budget() {
    let huge = Budget::new(u64::MAX);
    assert_eq!(huge.available(), u64::MAX);

    let small = Budget::new(1);
    assert_eq!(small.available(), 1);

    let zero = Budget::new(0);
    assert_eq!(zero.available(), 0);
}

#[test]
fn maxed_budget_still_obeys_affine_discipline() {
    let b = Budget::new(u64::MAX);
    
    let (parent, child) = b.split(1_000_000).expect("split should succeed");
    assert_eq!(parent.available(), u64::MAX - 1_000_000);
    assert_eq!(child.available(), 1_000_000);

    let (parent, _) = parent.spend(500, || ()).expect("spend should succeed");
    let _ = child.consume();
    let _ = parent.consume();
}

mod trusted_factory {
    use budget_spike::Budget;

    pub struct BudgetPolicy {
        pub max_budget_uc: u64,
        pub default_budget_uc: u64,
    }

    impl BudgetPolicy {
        pub fn mint(&self, amount_uc: u64) -> Result<Budget, &'static str> {
            if amount_uc > self.max_budget_uc {
                return Err("requested amount exceeds policy cap");
            }
            Ok(Budget::new(amount_uc))
        }
        
        pub fn mint_default(&self) -> Budget {
            Budget::new(self.default_budget_uc.min(self.max_budget_uc))
        }
    }
}

#[test]
fn trusted_factory_pattern_works() {
    use trusted_factory::BudgetPolicy;

    let policy = BudgetPolicy {
        max_budget_uc: 10_000_000, 
        default_budget_uc: 1_000_000, 
    };

    // Within-policy mint succeeds.
    let b = policy.mint(5_000_000).expect("within policy");
    assert_eq!(b.available(), 5_000_000);
    let _ = b.consume();

    let err = policy.mint(20_000_000).expect_err("over policy");
    assert_eq!(err, "requested amount exceeds policy cap");

    let b = policy.mint_default();
    assert_eq!(b.available(), 1_000_000);
    let _ = b.consume();
}

#[test]
fn adversarial_caller_can_bypass_trusted_factory() {
    let factory = trusted_factory::BudgetPolicy {
        max_budget_uc: 10_000_000,
        default_budget_uc: 1_000_000,
    };
    assert!(factory.mint(20_000_000).is_err());

    let bypass = Budget::new(20_000_000); 
    assert_eq!(bypass.available(), 20_000_000);
    let result = bypass.spend(20_000_001, || ());
    assert!(matches!(result, Err(BudgetError::Insufficient { .. })));
}

#[test]
fn conservation_holds_regardless_of_constructor_input() {
    let b = Budget::new(1_000_000);
    let (parent, child) = b.split(300_000).expect("split");
    let merged = parent.merge(child);
    assert_eq!(merged.available(), 1_000_000); 

    let b = Budget::new(u64::MAX / 2); // within Assumption A2
    let (parent, child) = b.split(1_000).expect("split");
    let merged = parent.merge(child);
    assert_eq!(merged.available(), u64::MAX / 2);
    let _ = merged.consume();
}