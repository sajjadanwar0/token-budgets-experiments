use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;
use tiktoken_rs::o200k_base;
use tokio::time::sleep;

#[derive(Parser)]
#[command(version, about = "A1 re-run on Anthropic")]
struct Cli {
    #[arg(long, default_value = "a1_rerun_results.csv")]
    output: String,

    #[arg(long, default_value = "claude-haiku-4-5")]
    model: String,

    #[arg(long, default_value = "150")]
    max_tokens: u32,

    #[arg(long, default_value = "10")]
    n_per_class: usize,

    #[arg(long, default_value = "300")]
    delay_ms: u64,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<Message<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
}

#[derive(Debug, Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize, Clone)]
struct ToolDef {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Serialize, Clone)]
struct Measurement {
    cell: String,
    prompt_id: usize,
    prompt_class: String,
    has_tools: bool,
    request_body_bytes: usize,
    tiktoken_estimator_tokens: usize,
    anthropic_input_tokens: u32,
    anthropic_output_tokens: u32,
    bt_ratio: f64,
    k_byte: f64,
    k_tiktoken: f64,
}

const PLAIN_SHORT: &[&str] = &[
    "What is the capital of France?",
    "Explain quantum entanglement in one paragraph.",
    "Define recursion in programming and give one small example.",
    "What is the speed of light in vacuum, and why is it constant?",
    "Name three planets in our solar system and one fact about each.",
    "Translate 'Hello, how are you today?' to French and Spanish.",
    "Compute 17 multiplied by 23 and show your work briefly.",
    "Define entropy in thermodynamics and explain why it tends to increase.",
    "Name a famous Renaissance painter and one of their works.",
    "What year did World War II end and what events marked its conclusion?",
];

const PLAIN_MEDIUM: &[&str] = &[
    "I'm trying to understand the difference between TCP and UDP protocols. Can you explain when I would use each one, what their main differences are, and provide some real-world examples of applications that use each protocol? Please cover reliability, ordering, head-of-line blocking, and modern alternatives like QUIC.",
    "Explain the concept of dependency injection in software engineering. What problem does it solve? How does it relate to inversion of control? Give a concrete example in a language of your choice, showing the same code with and without DI. Discuss when DI is overkill and when it pays off.",
    "Describe the structure of DNA in detail. What are nucleotides? How does base pairing work? What is the role of the double helix structure in DNA replication and protein synthesis? Cover the differences between DNA and RNA, and explain what happens during transcription and translation.",
    "Compare and contrast monolithic architecture with microservices. What are the trade-offs in terms of operational complexity, deployment, observability, and team autonomy? When would you choose one over the other? Provide examples of companies that have made transitions in either direction and what motivated those moves.",
    "Explain how virtual memory works in modern operating systems. What is the page table? What is the role of the TLB (Translation Lookaside Buffer)? How does demand paging differ from pre-paging? Discuss the impact of huge pages and what happens during a page fault.",
    "Discuss the Byzantine Generals problem in distributed systems. What is it? Why is it important? How do consensus algorithms like Paxos and Raft address related problems? What is the difference between CFT and BFT consensus, and where does PBFT fit in?",
    "Describe the process of photosynthesis in detail. What are the light-dependent and light-independent reactions? Where do they occur in the cell? What inputs and outputs are involved? Discuss the role of chlorophyll and the difference between C3 and C4 plants.",
    "Explain the difference between supervised and unsupervised learning in machine learning. Give examples of common algorithms in each category and describe a real-world application for each. Cover the role of labeled data, the curse of dimensionality, and when reinforcement learning is appropriate instead.",
    "Walk through how the SSL/TLS handshake works in TLS 1.3. What is the role of certificates? How is the symmetric key established? What attacks does this protect against, and what attacks does it not protect against? Compare to TLS 1.2 and explain why 0-RTT is controversial.",
    "Describe the concept of an eigenvalue and eigenvector in linear algebra. What is their geometric interpretation? Why are they important in applications like PCA (Principal Component Analysis), spectral graph theory, and quantum mechanics? Give a small worked example with a 2x2 matrix.",
];

const PLAIN_LONG: &[&str] = &[
    "I am writing a technical report on the evolution of distributed consensus algorithms from Lamport's original Paxos through Raft and into modern BFT systems used in blockchain consensus. I'd like a comprehensive explanation that covers: (1) the original Paxos paper and its three-phase protocol, including why it was considered notoriously difficult to implement correctly; (2) the Multi-Paxos optimization and how it amortizes the leader election cost across many decrees; (3) the Raft algorithm by Ongaro and Ousterhout, what they explicitly designed for understandability, and the three sub-problems (leader election, log replication, safety) they decomposed the problem into; (4) the FLP impossibility result and why every practical consensus algorithm sidesteps it through partial synchrony or randomization; (5) Byzantine fault tolerance starting with PBFT by Castro and Liskov, the view-change protocol, and the three-phase commit (pre-prepare, prepare, commit); (6) Tendermint and HotStuff as modern BFT consensus, particularly HotStuff's linear view-change complexity that made it suitable for permissionless settings; (7) Nakamoto consensus and the probabilistic finality of proof-of-work, contrasted with the deterministic finality of classical BFT; (8) modern hybrid approaches like Algorand, Casper FFG, and DiemBFT, including how they combine VRF-based leader election with BFT finality. Please make the explanation rigorous enough that someone with a graduate-level CS background but no distributed systems experience can follow it, while still touching on the cutting-edge research directions. Include comments on the practical implementation challenges (clock skew, network partitions, the gossip layer, signature aggregation in BFT, hardware requirements for the validator set). Cite the relevant seminal papers where appropriate but do not let citation density overwhelm the narrative.",
    "Write a tutorial-style explanation of how modern compilers perform register allocation, targeting a reader who has implemented a basic compiler (lexer, parser, AST, simple code generation) but has never gone deeper than that. Specifically cover: (1) the linear scan register allocation algorithm by Poletto and Sarkar, with its live interval representation and its O(n log n) running time; (2) the graph coloring approach pioneered by Chaitin and refined by Briggs, including chordal graph coloring and the spill heuristics; (3) SSA-based register allocation as used in modern compilers like LLVM, including how the SSA form changes the interference graph structure and enables polynomial-time optimal allocation in some cases; (4) the role of liveness analysis, how it is computed via iterative dataflow, and the difference between live-in and live-out sets; (5) coalescing of move instructions, Briggs vs. George coalescing heuristics, and why aggressive coalescing can cause spilling; (6) the impact of calling conventions, caller-saved vs. callee-saved registers, and how the allocator must respect these constraints; (7) handling of special-purpose registers (the stack pointer, frame pointer, condition codes), and how this interacts with the rest of allocation; (8) modern hardware concerns like NEON/AVX vector registers requiring separate allocation pools, register pairs on certain architectures, and the impact of out-of-order execution on what 'register pressure' even means. The goal is for the reader to come away able to implement a competent (not optimal) register allocator and to read the LLVM allocator source code with comprehension. Include small worked examples where they illuminate, but do not get bogged down in code; prefer prose with strategic diagrams.",
    "Provide a detailed comparison of the threading models in the major modern programming language runtimes. Cover at minimum: (1) Java's traditional thread-per-OS-thread model, the Thread class, the executor framework, and the recent virtual threads (Project Loom) and their integration with the existing synchronization primitives; (2) Go's goroutines and the M:N scheduler, the work-stealing approach, how goroutines interact with system calls, and the role of GOMAXPROCS; (3) Rust's async/await without runtime, the Future trait, why Rust deliberately stayed out of choosing a runtime, and the practical implications (tokio vs. async-std vs. smol vs. embassy for embedded); (4) Node.js's event loop and the libuv thread pool, what runs on the main thread vs. the pool, and how this interacts with native addons; (5) Python's GIL and the long history of attempts to remove it, including PEP 703 (free-threading), the practical implications for CPU-bound vs. I/O-bound workloads, and the subinterpreter approach (PEP 554); (6) Erlang/Elixir's BEAM and the actor model, the per-process heap, preemptive scheduling, and how this enables soft-real-time guarantees; (7) Kotlin coroutines and how they map to JVM threads via the dispatcher abstraction. For each runtime, explicitly state: what unit of concurrency the runtime exposes, how that unit is scheduled, what the cost (memory and CPU) of creating one is, and what shape of problem the runtime is good at vs. bad at. Then conclude with a comparison table.",
    "Write a thorough explanation of cache coherence protocols in multi-core systems. Start with the problem: why are cache coherence protocols necessary at all in a shared-memory multiprocessor? Explain the role of the memory consistency model and the relationship between cache coherence and consistency (they are related but distinct). Then walk through the MSI protocol (Modified, Shared, Invalid) as the simplest pedagogical model, showing the state transitions for read and write operations from each state, and the bus messages involved. Move on to MESI (adding Exclusive) and explain why Exclusive optimizes write-after-read patterns. Then MOESI (adding Owned) and how it enables cache-to-cache transfers without writing back to memory. Cover the directory-based alternative to snooping, used in systems with too many cores for a shared bus to scale, including how the directory tracks sharer sets and the trade-offs of full-map vs. limited-pointer directories. Discuss the relaxed memory models of x86 (TSO), ARM (weakly ordered), and the implications for memory barriers and atomic operations. Cover the false sharing problem, how cache lines (typically 64 bytes) become contention units, and the techniques to avoid it (padding, per-thread data, cache-line-aware data structures). Finally, discuss modern wrinkles: AMD's Zen NUMA topology with multiple memory controllers per package, Intel's mesh interconnect, and what happens when cache coherence becomes the scaling bottleneck (NUMA-aware allocation, RCU, lock-free algorithms).",
    "Explain in depth how garbage collection works in modern managed runtimes. Cover: (1) the basic dichotomy of tracing vs. reference counting, and why most modern systems use tracing (with reference counting as a sub-component or for specific objects); (2) the generational hypothesis and how it motivates young/old generation splits in collectors like Java's G1, Go's tricolor concurrent mark-sweep, and .NET's three-generation collector; (3) the tricolor invariant (white, gray, black) and how it formalizes the relationship between mutators and the collector during concurrent marking; (4) write barriers and read barriers, what each is used for (write barriers track inter-generational pointers; read barriers are used in some concurrent and incremental collectors like Shenandoah and ZGC), and the runtime cost they impose; (5) stop-the-world phases vs. fully concurrent phases, why some collectors retain a small STW phase for atomicity, and the practical implications for tail latency; (6) compaction strategies: copying collectors (semi-space, Cheney's algorithm), mark-compact, and the more sophisticated regional compaction in G1 and Shenandoah; (7) the trade-off between throughput and latency: parallel collectors optimize throughput by using all cores during a STW phase, while concurrent collectors optimize latency at the cost of write-barrier overhead and more complex synchronization; (8) GC tuning in practice: heap size, generation ratios, the role of escape analysis in reducing allocation pressure, and when to escape the GC entirely via off-heap allocation or value types. Aim for a reader who has used a managed language professionally but never read a GC paper.",
    "Write a detailed introduction to category theory aimed at a working software engineer who has heard the term 'monad' and 'functor' but doesn't really know what they mean beyond 'something to do with Haskell'. Cover: (1) what a category is formally (objects, morphisms, composition, identity, and the associativity and identity laws), and several concrete examples (Set, Top, Vect, Mon, and the category of types-and-functions in a programming language); (2) functors as structure-preserving maps between categories, with the canonical example of the List functor, and the laws (preservation of identity and composition); (3) natural transformations as morphisms between functors, what the naturality square is, and why naturality is the right notion of 'morphism between functors'; (4) the Yoneda lemma stated informally, why it's the 'fundamental theorem of category theory', and what it tells us about representable functors; (5) monads as monoid objects in the category of endofunctors, working through the unit and multiplication and how they correspond to return and join in Haskell; (6) the connection to programming: how Maybe, List, IO, State, Reader, and Writer are all monads, what their unit/join operations look like, and why monad transformers exist; (7) adjunctions, the unit-counit formulation, and how the free-forgetful adjunction generates the canonical monad on Set; (8) Kleisli categories as the formalism that links monads to ordinary function composition. Throughout, use code examples in Haskell (and occasionally OCaml or Scala for variety) to anchor the abstract definitions in something concrete. The goal is the reader can look at a monad-heavy library in Haskell and recognize the patterns, not become a research category theorist.",
    "Provide a rigorous walkthrough of how SAT solvers work in practice, from the basics through modern CDCL implementations. Cover: (1) the SAT problem itself, why it's the canonical NP-complete problem, and the practical importance (SAT solvers underpin most modern model checkers and synthesis tools); (2) the DPLL algorithm with backtracking, unit propagation, and pure literal elimination, including the standard pseudocode and a worked example; (3) the conflict-driven clause learning (CDCL) refinement that turned SAT from a theoretical curiosity into an industrial-strength tool: implication graphs, the first UIP cut, conflict analysis, and learned clause minimization; (4) variable selection heuristics: VSIDS (Variable State Independent Decaying Sum) and its modern variants like the LBD (Literal Block Distance) measure used by Glucose; (5) restart strategies and why they matter for escaping bad search trees: Luby sequences, geometric restarts, and adaptive restart policies; (6) the two-watched-literal data structure for efficient unit propagation, why two literals and not one, and how it interacts with backtracking; (7) clause database management: how learned clauses grow without bound and must be periodically pruned, and the heuristics for which clauses to keep; (8) preprocessing techniques: variable elimination, subsumption, blocked clause elimination, and how they can transform an instance to be much easier (or much harder); (9) the modern competitive solvers: MiniSat as the pedagogical baseline, Glucose, CaDiCaL, Kissat, and what each contributes; (10) extensions: incremental SAT for verification workflows, MaxSAT and pseudo-Boolean optimization, and the SMT extension via the lazy theory combination. Aim for someone who has implemented something graph-search-flavored before and wants to understand why SAT solvers are surprisingly fast in practice on instances that are theoretically intractable.",
    "Write a comprehensive comparison of the major modern container orchestration platforms and what design choices distinguish them. Cover at minimum: (1) Kubernetes as the de facto standard: the API server, etcd, controller manager, scheduler, kubelet, and kube-proxy architecture; the role of CRDs and the operator pattern; the StatefulSet vs. Deployment distinction; the service mesh layer (Istio, Linkerd); (2) HashiCorp Nomad: the simpler single-binary model, the lack of native networking abstractions, and why some shops prefer it for pre-existing infrastructure; (3) AWS ECS and EKS: the difference between the two, the role of Fargate in serverless containers, and the integration with VPC networking; (4) Google Cloud Run and Azure Container Apps as second-generation serverless container platforms: scale-to-zero, the request-driven autoscaler, and the trade-offs vs. always-on container deployments; (5) the Docker Swarm legacy and why it lost: the simpler model that turned out to be insufficient, and the implications for the lessons learned in K8s design; (6) the Linux primitives underlying all of these: namespaces, cgroups, capabilities, seccomp, the differences between runc, crun, and gVisor as runtimes; (7) networking: the CNI specification, the major implementations (Calico, Cilium, Weave), and how Cilium's eBPF-based dataplane changes the cost model; (8) storage: the CSI specification, why stateful workloads remained challenging for so long, and how operators like the postgres-operator or Vitess have tackled this; (9) security boundaries: container vs. VM, what gVisor and Kata Containers provide, and when you actually need them. The goal is the reader can evaluate whether their team should adopt Kubernetes or one of the alternatives, and understands the cost of running production K8s at organizational scale.",
    "Provide an exhaustive treatment of how modern compilers optimize for cache locality. Cover the underlying memory hierarchy first: registers, L1/L2/L3 caches with their typical sizes and latencies, the TLB, main memory, and the implications of NUMA on multi-socket systems. Then cover specific optimization passes: (1) loop interchange to traverse arrays in the order they are laid out in memory, including the dependency analysis required to know when interchange is legal; (2) loop tiling (also called blocking) to keep working sets fitting in cache, with the canonical matrix-multiply example showing tiled vs. naive; (3) array padding to avoid power-of-two strides that cause cache conflicts; (4) data layout transformations: array-of-structures vs. structure-of-arrays, when each is appropriate, and how this interacts with vectorization; (5) software prefetching: when the compiler inserts prefetch instructions, what the hardware prefetchers do automatically, and when manual prefetching helps; (6) inlining decisions: how the compiler trades off code size (worsening I-cache pressure) against the elimination of call overhead and the enabling of further optimizations; (7) basic block ordering for hot-path I-cache locality, including profile-guided optimization (PGO) and the use of branch prediction hints; (8) the impact of dead-code elimination and constant propagation on reducing the working set size; (9) the role of escape analysis in allowing stack allocation of objects that would otherwise be heap-allocated, with concomitant locality benefits. Discuss the polyhedral model as the unified framework for many of these loop transformations, and the practical compilers that implement substantial polyhedral analysis (Polly in LLVM, Pluto). End with a discussion of where current compilers fall short and what techniques (e.g., autotuning, ML-guided heuristics) are being explored.",
    "Walk through the design and operation of modern key-value stores at scale. Start with the basic problem: durably store key-value pairs, support get/put/delete, and scale beyond a single machine. Cover: (1) the LSM-tree (Log-Structured Merge) data structure used by RocksDB, LevelDB, Cassandra, and ScyllaDB: memtables, immutable SSTables, the role of WAL, and the compaction strategies (size-tiered, leveled, time-windowed) with their trade-offs; (2) the alternative B-tree-based stores (e.g., LMDB, BadgerDB v3+, MySQL's InnoDB): why they work better for read-heavy workloads, the COW vs. in-place update distinction, and the WAL strategy; (3) consistent hashing for distributing keys across nodes, the role of virtual nodes for load balancing, and how this interacts with replication; (4) replication strategies: primary-replica (Redis, MongoDB), leaderless (Dynamo, Cassandra) with quorum reads/writes, and consensus-based (etcd, FoundationDB); (5) the consistency spectrum from strong consistency (linearizability) through bounded staleness, prefix-consistency, and eventual consistency, with concrete examples of which production systems sit at each point; (6) the CAP theorem revisited: what it actually says (vs. the popularized version), PACELC, and why modern systems generally sacrifice availability for consistency under partition rather than the other way around; (7) global distribution: Spanner's TrueTime and external consistency, CockroachDB's HLC (Hybrid Logical Clock), and how these enable linearizable operations across geographies; (8) operational concerns: hot keys, range deletes, large value handling, schema evolution, backup and restore semantics, point-in-time recovery, and migration strategies. Aim for a reader who has used Redis and Postgres but never built a distributed database.",
];

fn make_tools() -> Vec<ToolDef> {
    use serde_json::json;
    vec![
        ToolDef {
            name: "execute_sql".to_string(),
            description: "Execute a SQL query against the production analytics database. The database contains tables for customers, orders, products, and inventory. Use this tool to answer business questions that require structured data lookup. Always check that the query is valid SQL before calling. The database is read-only; INSERT, UPDATE, DELETE will fail.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQL SELECT statement. Must use standard SQL syntax. Up to 1000 rows will be returned; use LIMIT for narrower queries."
                    },
                    "explain": {
                        "type": "boolean",
                        "description": "If true, returns the query execution plan instead of executing the query. Useful for debugging slow queries.",
                        "default": false
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "search_documents".to_string(),
            description: "Search the company's internal documentation, wikis, and knowledge bases. Returns up to 10 matching document excerpts ranked by relevance. The corpus includes engineering runbooks, product specifications, customer support articles, and meeting notes. Searches are case-insensitive and use semantic similarity in addition to keyword matching.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-form search query. Can be natural language ('how do I reset a customer's password') or keywords ('password reset workflow')."
                    },
                    "filter": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "enum": ["wiki", "runbook", "spec", "support", "meeting_notes"],
                                "description": "Restrict the search to a specific source corpus."
                            },
                            "date_after": {
                                "type": "string",
                                "format": "date",
                                "description": "ISO-8601 date; results updated before this date will be excluded."
                            }
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Maximum number of results to return."
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDef {
            name: "send_notification".to_string(),
            description: "Send a notification to a Slack channel, an email address, or a PagerDuty service. Use this tool only after you have collected and verified the information the user is asking about; notifications are visible to many people and should be accurate and concise. Notifications are rate-limited to 10 per minute per recipient.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Destination identifier. For Slack: '#channel-name' or '@user'. For email: a valid email address. For PagerDuty: a service key."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Short subject line (max 100 characters). Required for email; optional for Slack and PagerDuty."
                    },
                    "body": {
                        "type": "string",
                        "description": "The notification body. Supports limited Markdown for Slack. Plain text for email and PagerDuty."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "default": "normal",
                        "description": "Notification priority. 'urgent' will page the on-call engineer for PagerDuty channels."
                    }
                },
                "required": ["channel", "body"]
            }),
        },
    ]
}

const TOOL_PROMPTS: &[(&str, &str)] = &[
    // (class, user_prompt)
    ("sql_retry", "Find the top 5 customers by total revenue in the last 30 days. Include their customer ID, name, and total. Format the result as a markdown table."),
    ("sql_retry", "How many orders shipped to California addresses in Q3 2024, and what was the average order value? Show the breakdown by month."),
    ("sql_retry", "List all products that have less than 10 units in inventory and have been ordered at least once in the past week. Sort by units remaining ascending."),
    ("sql_retry", "What is the customer churn rate for customers who signed up in 2023, measured as the percentage who have not placed an order in the past 90 days?"),
    ("clarification", "I need to send something to the team about the deployment. Can you help me figure out what to do?"),
    ("clarification", "There's some information I'm looking for about onboarding. Can you check?"),
    ("clarification", "Could you look up the policy on this and let the right people know?"),
    ("clarification", "I want to update the team about an incident but I'm not sure who to notify."),
    ("arg_hallucination", "Send a high-priority message to the #engineering channel telling them the production database is being restarted in 5 minutes. Make sure on-call gets paged."),
    ("arg_hallucination", "Find the runbook for handling a payment-processor outage and then send the link to #ops-incident with a brief summary."),
];

fn tiktoken_count(s: &str) -> Result<usize> {
    let bpe = o200k_base().map_err(|e| anyhow::anyhow!("o200k_base load failed: {e}"))?;
    Ok(bpe.encode_with_special_tokens(s).len())
}

async fn one_call(
    client: &Client,
    api_key: &str,
    cli: &Cli,
    cell: &str,
    prompt_id: usize,
    prompt_class: &str,
    user_prompt: &str,
    tools: Option<&[ToolDef]>,
) -> Result<Measurement> {
    let request = AnthropicRequest {
        model: &cli.model,
        max_tokens: cli.max_tokens,
        messages: vec![Message {
            role: "user",
            content: user_prompt,
        }],
        system: None,
        tools,
    };
    let body_json = serde_json::to_string(&request)
        .context("failed to serialize request body")?;

    let request_body_bytes = body_json.len();
    let tiktoken_estimator_tokens = tiktoken_count(&body_json)?;

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(body_json)
        .send()
        .await
        .context("HTTP send failed")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Anthropic API error {status}: {text}");
    }
    let parsed: AnthropicResponse = resp.json().await
        .context("response JSON parse failed")?;

    let actual_in = parsed.usage.input_tokens;
    let actual_out = parsed.usage.output_tokens;

    let bt_ratio = request_body_bytes as f64 / actual_in.max(1) as f64;
    let k_byte = actual_in as f64 / request_body_bytes.max(1) as f64;
    let k_tiktoken = actual_in as f64 / tiktoken_estimator_tokens.max(1) as f64;

    Ok(Measurement {
        cell: cell.to_string(),
        prompt_id,
        prompt_class: prompt_class.to_string(),
        has_tools: tools.is_some(),
        request_body_bytes,
        tiktoken_estimator_tokens,
        anthropic_input_tokens: actual_in,
        anthropic_output_tokens: actual_out,
        bt_ratio,
        k_byte,
        k_tiktoken,
    })
}

async fn run_cell_1_plain(
    client: &Client,
    api_key: &str,
    cli: &Cli,
    out: &mut Vec<Measurement>,
) -> Result<()> {
    eprintln!("\n=== Cell 1: plain-text prompts, no tools ===");

    let mut id = 0;
    for (cls, corpus) in [
        ("plain_short",  PLAIN_SHORT),
        ("plain_medium", PLAIN_MEDIUM),
        ("plain_long",   PLAIN_LONG),
    ] {
        for i in 0..cli.n_per_class {
            let prompt = corpus[i % corpus.len()];
            id += 1;
            eprint!("  cell=1 id={:<3} class={:<12} ... ", id, cls);
            match one_call(client, api_key, cli, "cell_1_plain", id, cls, prompt, None).await {
                Ok(m) => {
                    eprintln!("bytes={} tokens={} bt={:.3} k_byte={:.3}",
                        m.request_body_bytes, m.anthropic_input_tokens, m.bt_ratio, m.k_byte);
                    out.push(m);
                }
                Err(e) => eprintln!("ERROR: {e}"),
            }
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
    }
    Ok(())
}

async fn run_cell_2_tools(
    client: &Client,
    api_key: &str,
    cli: &Cli,
    tools: &[ToolDef],
    out: &mut Vec<Measurement>,
) -> Result<()> {
    eprintln!("\n=== Cell 2: tool-augmented prompts (records both byte & tiktoken estimators) ===");

    let mut id = 0;
    // 10 distinct prompts in TOOL_PROMPTS; repeat across N runs
    let total = cli.n_per_class * 3;
    for i in 0..total {
        let (cls, prompt) = TOOL_PROMPTS[i % TOOL_PROMPTS.len()];
        id += 1;
        eprint!("  cell=2 id={:<3} class={:<18} ... ", id, cls);
        match one_call(client, api_key, cli, "cell_2_tools", id, cls, prompt, Some(tools)).await {
            Ok(m) => {
                eprintln!("bytes={} tikt={} actual={} bt={:.3} k_byte={:.3} k_tikt={:.3}",
                    m.request_body_bytes,
                    m.tiktoken_estimator_tokens,
                    m.anthropic_input_tokens,
                    m.bt_ratio, m.k_byte, m.k_tiktoken);
                out.push(m);
            }
            Err(e) => eprintln!("ERROR: {e}"),
        }
        sleep(Duration::from_millis(cli.delay_ms)).await;
    }
    Ok(())
}

fn summarize(measurements: &[Measurement]) {
    use std::collections::BTreeMap;
    let mut by_cell: BTreeMap<String, Vec<&Measurement>> = BTreeMap::new();
    for m in measurements {
        by_cell.entry(m.cell.clone()).or_default().push(m);
    }

    eprintln!("\n\n================== SUMMARY ==================\n");
    for (cell, ms) in &by_cell {
        let n = ms.len();
        if n == 0 { continue; }
        let bt_min = ms.iter().map(|m| m.bt_ratio).fold(f64::INFINITY, f64::min);
        let bt_max = ms.iter().map(|m| m.bt_ratio).fold(f64::NEG_INFINITY, f64::max);
        let bt_mean: f64 = ms.iter().map(|m| m.bt_ratio).sum::<f64>() / n as f64;
        let k_byte_max = ms.iter().map(|m| m.k_byte).fold(f64::NEG_INFINITY, f64::max);
        let k_tikt_max = ms.iter().map(|m| m.k_tiktoken).fold(f64::NEG_INFINITY, f64::max);
        let k_byte_mean: f64 = ms.iter().map(|m| m.k_byte).sum::<f64>() / n as f64;
        let k_tikt_mean: f64 = ms.iter().map(|m| m.k_tiktoken).sum::<f64>() / n as f64;
        let a1_holds = ms.iter().filter(|m| m.bt_ratio >= 1.0).count();
        let a1_holds_tikt = ms.iter().filter(|m| m.tiktoken_estimator_tokens >= m.anthropic_input_tokens as usize).count();

        eprintln!("Cell: {} (N={})", cell, n);
        eprintln!("  bt_ratio (bytes/actual_tokens):  min={:.3}  mean={:.3}  max={:.3}", bt_min, bt_mean, bt_max);
        eprintln!("  k_byte   (actual/bytes):         mean={:.3}  max={:.3}", k_byte_mean, k_byte_max);
        eprintln!("  k_tikt   (actual/tiktoken):      mean={:.3}  max={:.3}", k_tikt_mean, k_tikt_max);
        eprintln!("  A1 holds (bt_ratio >= 1):        {}/{}  ({:.1}%)", a1_holds, n, 100.0 * a1_holds as f64 / n as f64);
        eprintln!("  A1 holds w/ tiktoken estimator:  {}/{}  ({:.1}%)", a1_holds_tikt, n, 100.0 * a1_holds_tikt as f64 / n as f64);
        eprintln!();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY not set in environment")?;

    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    eprintln!("a1-rerun: Anthropic A1 (estimator soundness) characterization");
    eprintln!("model: {}", cli.model);
    eprintln!("max_tokens: {}", cli.max_tokens);
    eprintln!("n_per_class: {} (3 classes per cell = {} runs/cell, 2 cells = {} total)",
        cli.n_per_class, cli.n_per_class * 3, cli.n_per_class * 3 * 2);

    let tools = make_tools();
    let mut measurements: Vec<Measurement> = Vec::new();

    run_cell_1_plain(&client, &api_key, &cli, &mut measurements).await?;
    run_cell_2_tools(&client, &api_key, &cli, &tools, &mut measurements).await?;

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.output)
        .context("failed to open output CSV")?;
    for m in &measurements {
        wtr.serialize(m)?;
    }
    wtr.flush()?;
    eprintln!("\nWrote {} rows to {}", measurements.len(), cli.output);

    summarize(&measurements);

    Ok(())
}
