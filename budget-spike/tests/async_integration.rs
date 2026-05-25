use budget_spike::{
    Budget, BudgetError, CallError, call_with_budget,
    llm_client::MockClient,
};

#[tokio::test]
async fn mock_calls_within_budget() {
    let mut budget = Budget::new(100_000);
    let client = MockClient::sonnet_like();

    for i in 0..3 {
        let prompt = format!("call {i}");
        let (remaining, resp) = call_with_budget(&client, budget, &prompt, 200)
            .await
            .expect("should succeed");
        assert_eq!(resp.input_tokens, 50);
        assert_eq!(resp.output_tokens, 200);
        budget = remaining;
    }

    assert!(budget.available() < 100_000);
}

#[tokio::test]
async fn mock_terminates_cleanly_on_exhaustion() {
    let budget = Budget::new(100);
    let client = MockClient::sonnet_like();

    let result = call_with_budget(&client, budget, "test", 200).await;
    assert!(matches!(result, Err(CallError::Budget(BudgetError::Insufficient { .. }))));
}

#[tokio::test]
async fn split_across_spawn() {
    let parent = Budget::new(100_000);
    let (parent, child) = parent.split(40_000).expect("split should succeed");

    let handle: tokio::task::JoinHandle<Budget> = tokio::spawn(async move {
        let (child, _) = child.spend(30_000, || ()).expect("child spend in task");
        child 
    });

    let (parent, _) = parent.spend(50_000, || ()).expect("parent spend ok");

    let returned = handle.await.expect("task should complete");
    let merged = parent.merge(returned);
    
    assert_eq!(merged.available(), 20_000);
}

#[tokio::test]
async fn spend_then_await_then_spend() {
    let budget = Budget::new(50_000);
    let client = MockClient::sonnet_like();

    let (budget, _) = call_with_budget(&client, budget, "first", 100)
        .await
        .expect("first ok");

    let first_remaining = budget.available();

    let (budget, _) = call_with_budget(&client, budget, "second", 100)
        .await
        .expect("second ok");

    assert!(budget.available() < first_remaining);
}