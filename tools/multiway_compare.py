from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from operator import add
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError
from langgraph.errors import GraphRecursionError


PROVIDER_PRICING: Dict[str, Dict[str, float]] = {
    "openai": {
        "model": "gpt-4o",
        "input_per_token": 2.50 / 1_000_000,
        "output_per_token": 10.00 / 1_000_000,
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "input_per_token": 1.00 / 1_000_000,
        "output_per_token": 5.00 / 1_000_000,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "input_per_token": 0.59 / 1_000_000,
        "output_per_token": 0.79 / 1_000_000,
    },
    "mock": {
        "model": "MockSQLChatModel",
        "input_per_token": 0.15 / 1_000_000,
        "output_per_token": 0.60 / 1_000_000,
    },
}

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

@tool
def sql_query(query: str) -> str:
    return ""


@tool
def delete_record(id: str = "", name: str = "") -> str:
    return ""


@tool
def lookup_customer(name: str) -> str:
    """Look up a customer account by name."""
    return ""


WORKLOADS: Dict[str, Dict[str, Any]] = {
    "lang001": {
        "description": "SQL retry loop (LANG-001 reproduction)",
        "tool": sql_query,
        "tool_name": "sql_query",
        "tool_args": {"query": "SELECT * FRO users WHERE id=1"},
        "system_prompt": (
            "You are a database assistant. Use the sql_query tool. "
            "If it errors, try again."
        ),
        "user_prompt": "Find user with id=1 in the users table.",
        "tool_error": (
            "Error: SQL syntax error near 'FRO': invalid keyword. "
            "Did you mean 'FROM'? Please fix the query and retry."
        ),
    },
    "clarification": {
        "description": "Clarification loop on ambiguous tool description",
        "tool": delete_record,
        "tool_name": "delete_record",
        "tool_args": {"name": "report"},
        "system_prompt": (
            "You are a file management assistant. Use the delete_record "
            "tool to remove records. Ensure you delete the correct record."
        ),
        "user_prompt": "Please delete the report record.",
        "tool_error": (
            "Error: ambiguous record. Multiple records match. "
            "Please be more specific about which record to delete."
        ),
    },
    "arg_hallucination": {
        "description": "Hallucinated argument that the tool can't fulfil",
        "tool": lookup_customer,
        "tool_name": "lookup_customer",
        "tool_args": {"name": "Alice"},
        "system_prompt": (
            "You are a customer support assistant. Use lookup_customer to "
            "find customer accounts and answer questions about them."
        ),
        "user_prompt": "Look up the account for the customer named 'Alice'.",
        "tool_error": (
            "Error: customer 'Alice' not found. Tip: confirm the spelling "
            "matches an active account, or try lookup_customer again."
        ),
    },
}


class MockToolChatModel(BaseChatModel):
    growth_per_step: int = Field(default=60)
    base_input_tokens: int = Field(default=60)
    workload_tool_name: str = Field(default="sql_query")
    workload_tool_args: Dict[str, Any] = Field(default_factory=lambda: {"query": "SELECT * FRO users WHERE id=1"})

    @property
    def _llm_type(self) -> str:
        return "mock-tool-chat-model"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockToolChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        agent_turns = sum(1 for m in messages if isinstance(m, AIMessage))
        in_tok = self.base_input_tokens + self.growth_per_step * agent_turns
        out_tok = 40
        ai = AIMessage(
            content="",
            tool_calls=[{
                "name": self.workload_tool_name,
                "args": self.workload_tool_args,
                "id": f"call_{agent_turns}",
            }],
            usage_metadata={
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        )
        gen = ChatGeneration(message=ai)
        return ChatResult(
            generations=[gen],
            llm_output={
                "token_usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                },
                "model_name": "mock",
            },
        )


MockSQLChatModel = MockToolChatModel

def compute_cost_uc(in_tok: int, out_tok: int, provider: str) -> int:
    p = PROVIDER_PRICING[provider]
    cost_dollars = in_tok * p["input_per_token"] + out_tok * p["output_per_token"]
    return int(round(cost_dollars * 1_000_000))

class _AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]


def _build_langgraph(llm: Any, callback: Optional[BaseCallbackHandler],
                     guard_check: Optional[Callable[[], bool]],
                     guard_error_class: Optional[type],
                     workload: Optional[Dict[str, Any]] = None) -> Any:

    wl = workload or WORKLOADS["lang001"]
    tool_error_msg = wl["tool_error"]
    default_tool_name = wl["tool_name"]

    def agent_node(state: _AgentState) -> Dict[str, Any]:
        cb_list = [callback] if callback is not None else []
        response = llm.invoke(state["messages"], config={"callbacks": cb_list})
        if guard_check is not None and guard_check():
            raise guard_error_class("budget guard tripped")
        return {"messages": [response]}

    def tool_node(state: _AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        if not last.tool_calls:
            return {"messages": []}
        msgs = []
        for call in last.tool_calls:
            msgs.append(ToolMessage(
                content=tool_error_msg,
                tool_call_id=call["id"],
                name=call.get("name", default_tool_name),
            ))
        return {"messages": msgs}

    def should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool"
        return END

    g = StateGraph(_AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tool", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
    g.add_edge("tool", "agent")
    return g.compile()


@dataclass
class StepRecord:
    step: int
    input_tokens: int
    output_tokens: int
    cost_uc: int
    cumulative_uc: int


class CostTrackingCallback(BaseCallbackHandler):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.steps: List[StepRecord] = []
        self.cumulative_uc = 0

    def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
        in_tok, out_tok = self._extract_usage(response)
        cost_uc = compute_cost_uc(in_tok, out_tok, self.provider)
        self.cumulative_uc += cost_uc
        self.steps.append(StepRecord(
            step=len(self.steps) + 1,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_uc=cost_uc,
            cumulative_uc=self.cumulative_uc,
        ))

    def _extract_usage(self, response) -> tuple[int, int]:
        llm_out = getattr(response, "llm_output", None) or {}
        usage = llm_out.get("token_usage") or llm_out.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0)
                     or usage.get("input_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0)
                      or usage.get("output_tokens", 0))
        if in_tok == 0 and out_tok == 0:
            for gen_list in getattr(response, "generations", []):
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        meta = getattr(msg, "usage_metadata", None) or {}
                        in_tok = int(meta.get("input_tokens", 0))
                        out_tok = int(meta.get("output_tokens", 0))
                        if in_tok or out_tok:
                            break
                if in_tok or out_tok:
                    break
        return in_tok, out_tok


def _initial_messages(workload: Optional[Dict[str, Any]] = None) -> List[BaseMessage]:
    wl = workload or WORKLOADS["lang001"]
    return [
        SystemMessage(content=wl["system_prompt"]),
        HumanMessage(content=wl["user_prompt"]),
    ]


def run_langgraph_only(provider: str, cap_uc: int, growth: int,
                       recursion_limit: int,
                       workload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    wl = workload or WORKLOADS["lang001"]
    llm = _make_llm(provider, growth, wl)
    cb = CostTrackingCallback(provider)
    app = _build_langgraph(llm, cb, None, None, workload=wl)

    outcome: str
    try:
        app.invoke({"messages": _initial_messages(wl)},
                   config={"recursion_limit": recursion_limit})
        outcome = "completed_no_cap_hit"
    except GraphRecursionError:
        outcome = "structural_recursion_limit_hit"

    return _summarise(
        runtime="langgraph_only",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=cb.steps,
        cumulative_uc=cb.cumulative_uc,
    )

class BudgetExceededError(RuntimeError):
    """Raised by the RuntimeBudgetGuard when cumulative cost crosses the cap.
    The k-th call that crossed the threshold has already been billed."""


class RuntimeBudgetGuard(CostTrackingCallback):
    """AgentGuard-style runtime mitigation: subscribe to on_llm_end, deduct,
    trip a flag when cumulative >= cap. Architecturally, the guard fires
    AFTER the k-th call has been issued and billed."""

    def __init__(self, provider: str, cap_uc: int) -> None:
        super().__init__(provider)
        self.cap_uc = cap_uc
        self._tripped = False

    def is_tripped(self) -> bool:
        return self._tripped

    def on_llm_end(self, response, **kwargs) -> None:
        super().on_llm_end(response, **kwargs)
        if self.cumulative_uc >= self.cap_uc:
            self._tripped = True


def run_langgraph_with_guard(provider: str, cap_uc: int, growth: int,
                             recursion_limit: int,
                             workload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    wl = workload or WORKLOADS["lang001"]
    llm = _make_llm(provider, growth, wl)
    guard = RuntimeBudgetGuard(provider, cap_uc=cap_uc)
    app = _build_langgraph(llm, guard, guard.is_tripped, BudgetExceededError, workload=wl)

    outcome: str
    try:
        app.invoke({"messages": _initial_messages(wl)},
                   config={"recursion_limit": recursion_limit})
        outcome = "completed_no_cap_hit"
    except BudgetExceededError:
        outcome = "runtime_guard_fired_after_call"
    except GraphRecursionError:
        outcome = "structural_recursion_limit_hit"

    wasted_uc = guard.steps[-1].cost_uc if (
        outcome == "runtime_guard_fired_after_call" and guard.steps
    ) else 0

    summary = _summarise(
        runtime="langgraph_with_guard",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=guard.steps,
        cumulative_uc=guard.cumulative_uc,
    )
    summary["wasted_call_cost_uc"] = wasted_uc
    return summary

def run_crewai(provider: str, cap_uc: int, growth: int,
               max_iter: int) -> Dict[str, Any]:
    if provider == "mock":
        return _skipped(
            "crewai",
            reason="crewai_requires_live_provider_use_openai_anthropic_or_groq",
            cap_uc=cap_uc,
        )

    try:
        from crewai import Agent, Crew, Task
        from crewai.tools import BaseTool
    except ImportError:
        return _unavailable("crewai", "pip install crewai")

    class _SqlTool(BaseTool):
        name: str = "sql_query"
        description: str = ("Run a SQL query against the users table. "
                            "The users table has columns: id, name, email.")

        def _run(self, query: str) -> str:
            return ("Error: SQL syntax error near 'FRO': invalid keyword. "
                    "Did you mean 'FROM'? Please fix the query and retry.")

    model_str = {
        "openai": f"openai/{PROVIDER_PRICING['openai']['model']}",
        "anthropic": f"anthropic/{PROVIDER_PRICING['anthropic']['model']}",
        "groq": f"groq/{PROVIDER_PRICING['groq']['model']}",
    }[provider]

    # Verify the API key env var is set so we fail fast with a useful msg.
    _api_key_env(provider)

    agent = Agent(
        role="Database assistant",
        goal="Find the user with id=1 in the users table",
        backstory="You query the users table via the sql_query tool. "
                  "If the tool errors, fix the SQL and retry.",
        tools=[_SqlTool()],
        llm=model_str,
        max_iter=max_iter,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=("Find user with id=1 in the users table. "
                     "If sql_query errors, fix the SQL and retry."),
        expected_output="The user record or an error explanation.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    outcome = "completed_no_cap_hit"
    try:
        crew.kickoff()
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("max iter", "max_iter", "max iteration",
                                  "iteration limit", "maxiter")):
            outcome = "structural_max_iter_hit"
        else:
            raise

    in_tok, out_tok = _crewai_extract_usage(crew)
    total_uc = compute_cost_uc(in_tok, out_tok, provider)
    
    if outcome == "structural_max_iter_hit" and max_iter > 0 and (in_tok or out_tok):
        per_in = in_tok // max_iter
        per_out = out_tok // max_iter
        steps: List[StepRecord] = []
        cum = 0
        for i in range(max_iter):
            step_uc = compute_cost_uc(per_in, per_out, provider)
            cum += step_uc
            steps.append(StepRecord(
                step=i + 1,
                input_tokens=per_in,
                output_tokens=per_out,
                cost_uc=step_uc,
                cumulative_uc=cum,
            ))

        if steps:
            steps[-1] = StepRecord(
                step=steps[-1].step,
                input_tokens=in_tok - per_in * (max_iter - 1),
                output_tokens=out_tok - per_out * (max_iter - 1),
                cost_uc=total_uc - sum(s.cost_uc for s in steps[:-1]),
                cumulative_uc=total_uc,
            )
    elif in_tok or out_tok:
        steps = [StepRecord(
            step=1,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_uc=total_uc,
            cumulative_uc=total_uc,
        )]
    else:
        steps = []

    return _summarise(
        runtime="crewai",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=steps,
        cumulative_uc=total_uc,
    )


def _crewai_extract_usage(crew: Any) -> tuple[int, int]:
    usage = getattr(crew, "usage_metrics", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return (int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)))
    # Object form (newer CrewAI)
    in_tok = int(getattr(usage, "prompt_tokens", 0)
                 or getattr(usage, "input_tokens", 0))
    out_tok = int(getattr(usage, "completion_tokens", 0)
                  or getattr(usage, "output_tokens", 0))
    return in_tok, out_tok

def run_autogen(provider: str, cap_uc: int, growth: int,
                max_turns: int) -> Dict[str, Any]:
    if provider == "mock":
        return _skipped(
            "autogen",
            reason="autogen_requires_live_provider_use_openai_anthropic_or_groq",
            cap_uc=cap_uc,
        )

    try:
        from autogen import AssistantAgent, UserProxyAgent  # type: ignore
    except ImportError:
        return _unavailable("autogen", "pip install pyautogen>=0.2,<0.3")

    api_key = _api_key_env(provider)
    
    cfg = {
        "model": PROVIDER_PRICING[provider]["model"],
        "api_key": api_key,
    }
    if provider == "anthropic":
        cfg["api_type"] = "anthropic"
    elif provider == "groq":
        cfg["base_url"] = "https://api.groq.com/openai/v1"

    llm_config = {"config_list": [cfg], "cache_seed": None, "temperature": 0}

    assistant = AssistantAgent(
        name="db_assistant",
        system_message=("You are a database assistant. Use the sql_query "
                        "tool. The users table has columns: id, name, email. "
                        "If the tool errors, fix the SQL and retry."),
        llm_config=llm_config,
        max_consecutive_auto_reply=max_turns,
    )
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_turns,
        code_execution_config=False,
    )
    
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description="Run a SQL query against the users table")
    def sql_query_autogen(query: str) -> str:
        return ("Error: SQL syntax error near 'FRO': invalid keyword. "
                "Did you mean 'FROM'? Please fix the query and retry.")

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Find user with id=1 in the users table.",
        max_turns=max_turns,
    )

    in_tok, out_tok = _autogen_extract_usage(chat_result)
    total_uc = compute_cost_uc(in_tok, out_tok, provider)
    
    n_msgs = len(getattr(chat_result, "chat_history", []) or [])
    outcome = ("structural_max_turns_hit" if n_msgs >= max_turns * 2
               else "completed_no_cap_hit")

    if outcome == "structural_max_turns_hit" and max_turns > 0 and (in_tok or out_tok):
        per_in = in_tok // max_turns
        per_out = out_tok // max_turns
        steps: List[StepRecord] = []
        cum = 0
        for i in range(max_turns):
            step_uc = compute_cost_uc(per_in, per_out, provider)
            cum += step_uc
            steps.append(StepRecord(
                step=i + 1, input_tokens=per_in, output_tokens=per_out,
                cost_uc=step_uc, cumulative_uc=cum,
            ))
        if steps:
            steps[-1] = StepRecord(
                step=steps[-1].step,
                input_tokens=in_tok - per_in * (max_turns - 1),
                output_tokens=out_tok - per_out * (max_turns - 1),
                cost_uc=total_uc - sum(s.cost_uc for s in steps[:-1]),
                cumulative_uc=total_uc,
            )
    elif in_tok or out_tok:
        steps = [StepRecord(
            step=1, input_tokens=in_tok, output_tokens=out_tok,
            cost_uc=total_uc, cumulative_uc=total_uc,
        )]
    else:
        steps = []

    return _summarise(
        runtime="autogen",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=steps,
        cumulative_uc=total_uc,
    )


def _autogen_extract_usage(chat_result: Any) -> tuple[int, int]:
    """Pull aggregate prompt/completion tokens from AutoGen's chat_result.cost."""
    cost = getattr(chat_result, "cost", None)
    if not cost:
        return 0, 0
    
    bucket = cost.get("usage_excluding_cached_inference") or cost
    in_tok = 0
    out_tok = 0
    for key, val in bucket.items():
        if key == "total_cost" or not isinstance(val, dict):
            continue
        in_tok += int(val.get("prompt_tokens", 0))
        out_tok += int(val.get("completion_tokens", 0))
    return in_tok, out_tok

def run_token_capabilities(provider: str, cap_uc: int, growth: int,
                           recursion_limit: int,
                           workload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
    wl = workload or WORKLOADS["lang001"]
    llm = _make_llm(provider, growth, wl)
    cb = CostTrackingCallback(provider)
    remaining = cap_uc
    messages = _initial_messages(wl)

    outcome: str = "completed_no_cap_hit"
    for step_idx in range(1, recursion_limit // 2 + 1):
        agent_turns = sum(1 for m in messages if isinstance(m, AIMessage))
        est_input = 60 + growth * agent_turns
        est_output = 40
        est_uc = compute_cost_uc(est_input, est_output, provider)

        if est_uc > remaining:
            outcome = "compile_time_reservation_refused"
            break

        remaining -= est_uc
        ai = llm.invoke(messages, config={"callbacks": [cb]})
        messages.append(ai)
        if isinstance(ai, AIMessage) and ai.tool_calls:
            messages.append(ToolMessage(
                content=wl["tool_error"],
                tool_call_id=ai.tool_calls[0]["id"],
                name=wl["tool_name"],
            ))

    return _summarise(
        runtime="token_capabilities",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=cb.steps,
        cumulative_uc=cb.cumulative_uc,
    )

def _make_llm(provider: str, growth: int, workload: Optional[Dict[str, Any]] = None) -> Any:
    wl = workload or WORKLOADS["lang001"]
    if provider == "mock":
        return MockToolChatModel(
            growth_per_step=growth,
            workload_tool_name=wl["tool_name"],
            workload_tool_args=wl["tool_args"],
        ).bind_tools([wl["tool"]])
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=PROVIDER_PRICING["openai"]["model"],
            temperature=0,
        ).bind_tools([wl["tool"]])
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=PROVIDER_PRICING["anthropic"]["model"],
            temperature=0,
        ).bind_tools([wl["tool"]])
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=PROVIDER_PRICING["groq"]["model"],
            temperature=0,
        ).bind_tools([wl["tool"]])
    raise ValueError(f"unknown provider: {provider}")


def _api_key_env(provider: str) -> str:
    import os
    var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }[provider]
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"environment variable {var} not set")
    return val


def _summarise(runtime: str, outcome: str, cap_uc: int,
               cb_steps: List[StepRecord], cumulative_uc: int) -> Dict[str, Any]:
    overshoot_uc = max(0, cumulative_uc - cap_uc)
    undershoot_uc = max(0, cap_uc - cumulative_uc) if "structural" in outcome else 0
    pct_of_cap = (cumulative_uc / cap_uc * 100.0) if cap_uc > 0 else 0.0
    return {
        "runtime": runtime,
        "outcome": outcome,
        "agent_steps": len(cb_steps),
        "cap_uc": cap_uc,
        "total_spent_uc": cumulative_uc,
        "pct_of_cap": round(pct_of_cap, 2),
        "overshoot_uc": overshoot_uc,
        "structural_undershoot_uc": undershoot_uc,
        "wasted_call_cost_uc": 0,  # overridden by guard
        "per_step": [asdict(s) for s in cb_steps],
    }


def _unavailable(name: str, install_hint: str) -> Dict[str, Any]:
    return {
        "runtime": name,
        "outcome": f"unavailable_install_with_{install_hint.replace(' ', '_')}",
        "agent_steps": 0,
        "cap_uc": 0,
        "total_spent_uc": 0,
        "pct_of_cap": 0.0,
        "overshoot_uc": 0,
        "structural_undershoot_uc": 0,
        "wasted_call_cost_uc": 0,
        "per_step": [],
    }


def _skipped(name: str, reason: str, cap_uc: int) -> Dict[str, Any]:
    """Recorded skip — runtime not applicable to current provider, not an error."""
    return {
        "runtime": name,
        "outcome": f"skipped_{reason}",
        "agent_steps": 0,
        "cap_uc": cap_uc,
        "total_spent_uc": 0,
        "pct_of_cap": 0.0,
        "overshoot_uc": 0,
        "structural_undershoot_uc": 0,
        "wasted_call_cost_uc": 0,
        "per_step": [],
    }

class LiteLLMBudgetCallback(CostTrackingCallback):
    def __init__(self, provider: str, cap_uc: int) -> None:
        super().__init__(provider)
        from litellm import BudgetManager
        self.user_id = "tc_eval_user"
        self.bm = BudgetManager(project_name="tc_eval_project")
        self.cap_dollars = cap_uc / 1_000_000
        self.bm.create_budget(total_budget=self.cap_dollars, user=self.user_id)
        self._tripped = False

    def is_tripped(self) -> bool:
        return self._tripped

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)

        try:
            from litellm import completion_cost
            generations = getattr(response, "generations", None) or []
            for batch in generations:
                for gen in batch:
                    msg = getattr(gen, "message", None)
                    if msg is None:
                        continue
                    usage = getattr(msg, "usage_metadata", None) or {}
                    in_tok = int(usage.get("input_tokens", 0))
                    out_tok = int(usage.get("output_tokens", 0))
                    # Compute call cost via litellm with the same model name
                    model = PROVIDER_PRICING[self.provider]["model"]
                    cost = completion_cost(
                        model=model,
                        prompt_tokens=in_tok,
                        completion_tokens=out_tok,
                    )
                    current = self.bm.get_current_cost(user=self.user_id)
                    self.bm.update_cost(
                        completion_obj=None,
                        user=self.user_id,
                        model=model,
                        input_text="",  # we already have token counts
                        output_text="",
                    )
                    new_total = current + cost
                    self.bm.user_dict[self.user_id]["current_cost"] = new_total
                    if new_total >= self.cap_dollars:
                        self._tripped = True
        except Exception as e:
            if self.cumulative_uc >= self.cap_uc_target:
                self._tripped = True

    @property
    def cap_uc_target(self) -> int:
        return int(self.cap_dollars * 1_000_000)


def run_litellm_proxy(provider: str, cap_uc: int, growth: int,
                      recursion_limit: int,
                      workload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        import litellm 
    except ImportError:
        return {
            "runtime": "litellm_proxy",
            "outcome": "litellm_not_installed",
            "agent_steps": 0,
            "cap_uc": cap_uc,
            "total_spent_uc": 0,
            "pct_of_cap": 0.0,
            "overshoot_uc": 0,
            "structural_undershoot_uc": cap_uc,
            "wasted_call_cost_uc": 0,
            "per_step": [],
        }

    wl = workload or WORKLOADS["lang001"]
    llm = _make_llm(provider, growth, wl)
    cb = LiteLLMBudgetCallback(provider, cap_uc=cap_uc)
    app = _build_langgraph(llm, cb, cb.is_tripped, BudgetExceededError, workload=wl)

    outcome: str
    try:
        app.invoke({"messages": _initial_messages(wl)},
                   config={"recursion_limit": recursion_limit})
        outcome = "completed_no_cap_hit"
    except BudgetExceededError:
        outcome = "litellm_budget_exceeded_after_call"
    except GraphRecursionError:
        outcome = "structural_recursion_limit_hit"

    wasted_uc = cb.steps[-1].cost_uc if (
        outcome == "litellm_budget_exceeded_after_call" and cb.steps
    ) else 0

    summary = _summarise(
        runtime="litellm_proxy",
        outcome=outcome,
        cap_uc=cap_uc,
        cb_steps=cb.steps,
        cumulative_uc=cb.cumulative_uc,
    )
    summary["wasted_call_cost_uc"] = wasted_uc
    return summary

RUNTIMES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "langgraph_only": run_langgraph_only,
    "langgraph_with_guard": run_langgraph_with_guard,
    "crewai": run_crewai,
    "autogen": run_autogen,
    "token_capabilities": run_token_capabilities,
    "litellm_proxy": run_litellm_proxy,
}


def main() -> int:
    p = argparse.ArgumentParser(
        description="5-way runtime head-to-head on LANG-001 reproduction."
    )
    p.add_argument("--runs", type=int, default=10,
                   help="N runs per runtime (default 10)")
    p.add_argument("--cap-uc", type=int, default=540,
                   help="Budget cap in micro-cents (default 540 = $0.00054, "
                        "matches the LANG-001 cap used in Section 5.3 of the paper)")
    p.add_argument("--growth", type=int, default=60,
                   help="Per-step input-token growth (default 60)")
    p.add_argument("--provider", type=str, default="mock",
                   choices=["mock", "openai", "anthropic", "groq"],
                   help="LLM provider (default mock; live providers cost money)")
    p.add_argument("--runtimes", type=str, default=",".join(RUNTIMES.keys()),
                   help="Comma-separated subset of runtimes to run")
    p.add_argument("--recursion-limit", type=int, default=20,
                   help="LangGraph recursion_limit (default 20 = 10 agent invocations)")
    p.add_argument("--max-iter", type=int, default=5,
                   help="CrewAI max_iter (default 5)")
    p.add_argument("--max-turns", type=int, default=4,
                   help="AutoGen max_turns (default 4)")
    p.add_argument("--workload", type=str, default="lang001",
                   choices=list(WORKLOADS.keys()),
                   help="Workload selection: lang001 (SQL retry, default), "
                        "clarification (ambiguous-tool clarification loop), "
                        "or arg_hallucination (hallucinated argument loop)")
    p.add_argument("--output-csv", type=str, default=None,
                   help="Write per-run rows to this CSV")
    p.add_argument("--json-detail", type=str, default=None,
                   help="Write full per-step ledgers to this JSON")
    args = p.parse_args()

    selected = [r.strip() for r in args.runtimes.split(",") if r.strip()]
    for r in selected:
        if r not in RUNTIMES:
            print(f"unknown runtime: {r}; valid: {list(RUNTIMES)}", file=sys.stderr)
            return 2

    workload = WORKLOADS[args.workload]
    rows: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []

    print(f"5-way runtime head-to-head | provider={args.provider} | "
          f"workload={args.workload} ({workload['description']}) | "
          f"cap=${args.cap_uc/1_000_000:.6f} | growth={args.growth} | "
          f"runs={args.runs}")
    print()

    for runtime_name in selected:
        runner = RUNTIMES[runtime_name]
        for run_id in range(1, args.runs + 1):
            t0 = time.time()
            kwargs = {
                "provider": args.provider,
                "cap_uc": args.cap_uc,
                "growth": args.growth,
            }
            if runtime_name in ("langgraph_only", "langgraph_with_guard",
                                "token_capabilities", "litellm_proxy"):
                kwargs["recursion_limit"] = args.recursion_limit
                kwargs["workload"] = workload
            elif runtime_name == "crewai":
                kwargs["max_iter"] = args.max_iter
          
            elif runtime_name == "autogen":
                kwargs["max_turns"] = args.max_turns

            try:
                result = runner(**kwargs)
            except Exception as e:
                print(f"  {runtime_name} run {run_id}: ERROR {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            row = {
                "runtime": result["runtime"],
                "run_id": run_id,
                "provider": args.provider,
                "workload": args.workload,
                "outcome": result["outcome"],
                "agent_steps": result["agent_steps"],
                "cap_uc": result["cap_uc"],
                "total_spent_uc": result["total_spent_uc"],
                "pct_of_cap": result["pct_of_cap"],
                "overshoot_uc": result["overshoot_uc"],
                "structural_undershoot_uc": result["structural_undershoot_uc"],
                "wasted_call_cost_uc": result["wasted_call_cost_uc"],
                "wall_seconds": round(elapsed, 3),
            }
            rows.append(row)
            detail.append({**row, "per_step": result["per_step"]})

    # Console summary
    print()
    print("=" * 110)
    print(f"{'runtime':<24} {'outcome':<38} {'steps':>6} {'spent_uc':>10} "
          f"{'pct_cap':>8} {'overshoot':>10} {'undershoot':>11}")
    print("=" * 110)
    for runtime_name in selected:
        rt_rows = [r for r in rows if r["runtime"] == runtime_name]
        if not rt_rows:
            continue
        ex = rt_rows[0]
        avg_spent = sum(r["total_spent_uc"] for r in rt_rows) / len(rt_rows)
        avg_pct = sum(r["pct_of_cap"] for r in rt_rows) / len(rt_rows)
        avg_over = sum(r["overshoot_uc"] for r in rt_rows) / len(rt_rows)
        avg_under = sum(r["structural_undershoot_uc"] for r in rt_rows) / len(rt_rows)
        print(f"{runtime_name:<24} {ex['outcome']:<38} {ex['agent_steps']:>6} "
              f"{avg_spent:>10.0f} {avg_pct:>7.2f}% {avg_over:>10.0f} "
              f"{avg_under:>11.0f}")

    print()
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output_csv}")

    if args.json_detail:
        with open(args.json_detail, "w") as f:
            json.dump(detail, f, indent=2)
        print(f"wrote per-step detail to {args.json_detail}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
