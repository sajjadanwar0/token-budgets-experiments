from __future__ import annotations
import os
import sys
import traceback

print("=" * 60)
print("AGENT CONTRACTS DIAGNOSTIC")
print("=" * 60)

print("\n[1] Importing ai-agent-contracts...")
try:
    import agent_contracts
    print(f"    OK: agent_contracts {agent_contracts.__version__ if hasattr(agent_contracts, '__version__') else '(no version attr)'}")
    print(f"    location: {agent_contracts.__file__}")
except Exception as e:
    print(f"    FAIL: {e}")
    sys.exit(1)

try:
    from agent_contracts import (
        Contract, ContractedLLM, ResourceConstraints, ContractMode
    )
    print(f"    OK: Contract, ContractedLLM, ResourceConstraints, ContractMode")
except Exception as e:
    print(f"    FAIL importing names: {e}")
    sys.exit(1)

print("\n[2] Importing litellm...")
try:
    import litellm
    print(f"    OK: litellm {litellm.__version__ if hasattr(litellm, '__version__') else '(no version attr)'}")
except Exception as e:
    print(f"    FAIL: {e}")
    sys.exit(1)

# 3. API key
print("\n[3] Checking ANTHROPIC_API_KEY...")
key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"    {'set, prefix=' + key[:12] + '...' if key else 'NOT SET'}")

print("\n[4] Inspecting ContractedLLM class...")
print(f"    type: {type(ContractedLLM)}")
print(f"    __init__ signature: {ContractedLLM.__init__.__doc__ if ContractedLLM.__init__.__doc__ else '(no docstring)'}")
print(f"    public methods: {[m for m in dir(ContractedLLM) if not m.startswith('_')]}")

print("\n[5] Building contract...")
try:
    contract = Contract(
        id="diag-test",
        name="diagnostic",
        mode=ContractMode.BALANCED,
        resources=ResourceConstraints(
            tokens=1000,
            api_calls=10,
            cost_usd=0.01,
        ),
    )
    print(f"    OK: contract.id={contract.id}")
    print(f"    contract.state = {contract.state}")
    print(f"    contract attrs = {[a for a in dir(contract) if not a.startswith('_')]}")
except Exception as e:
    print(f"    FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[6] Trying contract.activate() if available...")
if hasattr(contract, "activate"):
    try:
        contract.activate()
        print(f"    OK: contract.state = {contract.state}")
    except Exception as e:
        print(f"    activate() raised: {e}")
else:
    print("    no activate() method present")

print("\n[7] Trying ContractedLLM(contract) context manager...")
try:
    llm = ContractedLLM(contract)
    print(f"    instance created: type={type(llm).__name__}")
    print(f"    instance attrs: {[a for a in dir(llm) if not a.startswith('_')]}")
    print(f"    has __enter__: {hasattr(llm, '__enter__')}")
    print(f"    has completion: {hasattr(llm, 'completion')}")
except Exception as e:
    print(f"    FAIL on instantiation: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[8] Entering with-block and calling completion()...")
try:
    with ContractedLLM(contract) as llm:
        print(f"    inside with-block: type(llm)={type(llm).__name__}")
        print(f"    llm methods: {[m for m in dir(llm) if not m.startswith('_')][:15]}")

        for model_name in [
            "claude-sonnet-4-5-20250929",
            "anthropic/claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20241022",
        ]:
            print(f"\n    trying model={model_name!r}...")
            try:
                response = llm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": "Reply with the single word 'pong'."}],
                    max_tokens=10,
                    temperature=0,
                )
                print(f"      OK: type(response)={type(response).__name__}")
                content = None
                try:
                    content = response.choices[0].message.content
                except Exception:
                    pass
                print(f"      content: {content!r}")
                print(f"      usage: {getattr(response, 'usage', None)}")
                print(f"    SUCCESS with model={model_name}")
                break
            except Exception as e:
                print(f"      FAIL: {type(e).__name__}: {e}")
        else:
            print(f"\n    ALL model names failed")
except Exception as e:
    print(f"    FAIL on __enter__: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n[9] Done. Paste this whole output back to Claude.")