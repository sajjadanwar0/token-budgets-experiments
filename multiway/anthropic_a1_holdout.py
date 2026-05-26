import os
import csv
import sys
import math

try:
    import anthropic
except ImportError:
    sys.exit("Install: pip install anthropic")

MODEL = "claude-haiku-4-5-20251001"
MARGIN = 2.0
OUTPUT_CSV = "a1_holdout_results.csv"

CJK_PROMPTS = [
    "古之欲明明德於天下者,先治其國;欲治其國者,先齊其家;欲齊其家者,先修其身;欲修其身者,先正其心;欲正其心者,先誠其意;欲誠其意者,先致其知;致知在格物。",
    "天行健,君子以自強不息;地勢坤,君子以厚德載物。雲行雨施,品物流形。大明終始,六位時成,時乘六龍以御天。",
    "春過ぎて夏来るらし白妙の衣干したり天の香具山。我が背子と二人見ませばいくばくか此の降る雪の嬉しからまし。",
    "プログラミング言語Rustは2010年にGraydon Hoareが設計し、その後Mozillaが開発を引き継ぎました。所有権システム(ownership system)が特徴です。",
    "한글은 1443년 세종대왕에 의해 창제되었으며, 1446년에 훈민정음이라는 이름으로 반포되었다. 자음 14개와 모음 10개로 구성된 표음문자이다.",
    "수신제가치국평천하(修身齊家治國平天下)는 대학(大學)에 나오는 유교의 핵심 가르침으로, 자신의 몸을 닦고 집안을 가지런히 한 후에야 나라를 다스리고 천하를 평정할 수 있다는 뜻이다.",
    "深度学习(Deep Learning)模型如GPT-4使用Transformer架构,其核心是注意力机制(Attention Mechanism)。Token化过程将文本分割为子词(subword)单元。",
    "Tiếng Việt là ngôn ngữ chính thức của Việt Nam, được hơn 90 triệu người bản ngữ sử dụng. Hệ thống chữ viết hiện đại dựa trên bảng chữ cái Latin với các dấu phụ.",
    "اللغة العربية هي إحدى أكثر اللغات السامية انتشاراً في العالم، يتحدث بها أكثر من 422 مليون نسمة وتستخدم كلغة رسمية في 26 دولة.",
    "הָאָדָם נוֹלָד חָפְשִׁי וְשָׁוֶה בִּכְבוֹדוֹ וּבִזְכֻיּוֹתָיו. כֻּלָּם חוֹנְנוּ בְּתְבוּנָה וּבְמַצְפּוּן.",
    "भारत एक दक्षिण एशियाई देश है। यह क्षेत्रफल की दृष्टि से विश्व का सातवाँ बड़ा एवं जनसंख्या की दृष्टि से सबसे बड़ा देश है। राजधानी नई दिल्ली है।",
    "Программирование на Rust требует понимания системы владения. Каждое значение имеет владельца, и когда владелец выходит из области видимости, значение освобождается.",
    "Η Ελληνική γλώσσα είναι μία από τις παλαιότερες ινδοευρωπαϊκές γλώσσες με συνεχή προφορική και γραπτή παράδοση από τον 14ο αιώνα π.Χ. μέχρι σήμερα.",
    "ภาษาไทยเป็นภาษาราชการของประเทศไทย เป็นภาษาในตระกูลภาษาขร้า-ไท และมีระบบการเขียนที่ใช้อักษรไทยซึ่งดัดแปลงมาจากอักษรเขมรโบราณ",
    "中国Rust社区:こんにちは Здравствуйте مرحبا नमस्ते 안녕하세요 Γειά σας שלום สวัสดี",
]

CODE_PROMPTS = [
    """async def fetch_user_data(user_id: int, db: Database) -> Optional[User]:
    async with db.connection() as conn:
        result = await conn.execute(
            "SELECT * FROM users WHERE id = $1 AND deleted_at IS NULL",
            user_id
        )
        row = await result.fetchone()
        return User(**row) if row else None""",
    """fn process<T: Iterator<Item = u32>>(iter: T, budget: Budget) -> Result<Vec<u32>, BudgetError> {
    let collected: Vec<u32> = iter.collect();
    let cost = collected.len() as u64 * 4;
    let _ = budget.spend(cost)?;
    Ok(collected.into_iter().map(|x| x * 2).collect())
}""",
    """function useBudgetTracker(initialBudget) {
  const [budget, setBudget] = useState(initialBudget);
  const [history, setHistory] = useState([]);

  const spend = useCallback((amount) => {
    if (amount > budget) throw new Error(`Insufficient: need ${amount}, have ${budget}`);
    setBudget(b => b - amount);
    setHistory(h => [...h, { amount, timestamp: Date.now() }]);
  }, [budget]);

  return { budget, spend, history };
}""",
    """processRequest :: (MonadIO m, MonadReader Config m, MonadError AppError m) 
               => Request 
               -> m Response
processRequest req = do
  cfg <- ask
  validated <- validateRequest req `catchError` (throwError . ValidationFailed)
  result <- liftIO $ executeQuery (dbConfig cfg) (toQuery validated)
  pure $ Response { status = 200, body = toJSON result }""",
    """WITH recursive budget_tree AS (SELECT id, parent_id, name, allocated_cents, 0 as depth
                                      FROM budgets
                                      WHERE parent_id IS NULL
                                      UNION ALL
                                      SELECT b.id, b.parent_id, b.name, b.allocated_cents, bt.depth + 1
                                      FROM budgets b
                                               JOIN budget_tree bt ON b.parent_id = bt.id
                                      WHERE bt.depth < 10)
       SELECT *
       FROM budget_tree
       WHERE allocated_cents > 1000
       ORDER BY depth, name;""",
    """const PATTERNS = {
  email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/,
  ipv4: /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/,
  uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  semver: /^(\\d+)\\.(\\d+)\\.(\\d+)(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?$/,
};""",
    """#!/bin/bash
set -euo pipefail
for env in prod staging dev; do
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: budget-config
  namespace: ${env}
data:
  daily_cap_uc: "$((env == 'prod' ? 1000000 : 100000))"
  margin_ratio: "2.0"
EOF
done""",
    """module budget_tracker #(parameter WIDTH = 32) (
    input wire clk,
    input wire rst_n,
    input wire [WIDTH-1:0] spend_amt,
    input wire spend_valid,
    output reg [WIDTH-1:0] balance,
    output reg overshoot
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin balance <= 0; overshoot <= 0; end
        else if (spend_valid) begin
            if (spend_amt > balance) overshoot <= 1;
            else balance <= balance - spend_amt;
        end
    end
endmodule""",
    """budget_spend:
    pushq   %rbp
    movq    %rsp, %rbp
    movq    %rdi, -8(%rbp)      # budget ptr
    movq    %rsi, -16(%rbp)     # amount
    movq    -8(%rbp), %rax
    movq    (%rax), %rcx        # load balance
    cmpq    -16(%rbp), %rcx
    jb      .Loverflow
    subq    -16(%rbp), %rcx
    movq    %rcx, (%rax)
    xorl    %eax, %eax
    jmp     .Lreturn
.Loverflow:
    movl    $1, %eax
.Lreturn:
    popq    %rbp
    ret""",
    """(define (budget-spend! budget amount)
  (cond ((negative? amount)
         (error 'invalid-amount "amount must be non-negative" amount))
        ((> amount (budget-balance budget))
         (raise (make-budget-exceeded budget amount)))
        (else
         (set-budget-balance! budget
           (- (budget-balance budget) amount))
         (cons 'ok (budget-balance budget)))))""",
]

MATH_PROMPTS = [
    r"""Lemma 1 (cap-soundness): Let $\mathcal{B} = (b_0, S, T)$ be a budget instance where 
$S \subseteq T$ is the spent set and $T$ the total trace. Define $\sigma(t) = \sum_{s \in S, s \le t} c(s)$. 
Then $\sigma(t) \le b_0$ for all $t \in T$, conditional on Assumption A1: 
$\forall p, E_P(p) \ge \mathrm{billable\_tokens}_P(p)$.""",
    r"""\begin{align}
\mathcal{L}(\theta) &= -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i) + \lambda \|\theta\|_2^2 \\
\nabla_\theta \mathcal{L} &= -\frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \log p_\theta(y_i \mid x_i) + 2\lambda\theta \\
\theta_{t+1} &= \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t)
\end{align}""",
    r"""The Bellman equation for the value function: $V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s\right] = \sum_a \pi(a|s) \left[r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$ where $\gamma \in [0,1)$ is the discount factor.""",
    r"""Theorem (FTC): If $f: [a,b] \to \mathbb{R}$ is continuous and $F(x) = \int_a^x f(t)\,dt$, then $F$ is differentiable on $(a,b)$ and $F'(x) = f(x)$. Corollary: $\int_a^b f(x)\,dx = F(b) - F(a)$ for any antiderivative $F$ of $f$.""",
    r"""Cauchy-Schwarz: $|\langle u, v \rangle|^2 \le \langle u, u \rangle \cdot \langle v, v \rangle$. Triangle inequality: $\|u + v\| \le \|u\| + \|v\|$. In $\mathbb{R}^n$: $\left(\sum_{i=1}^n u_i v_i\right)^2 \le \left(\sum_{i=1}^n u_i^2\right)\left(\sum_{i=1}^n v_i^2\right)$.""",
    r"""$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_p \frac{1}{1 - p^{-s}}$ for $\Re(s) > 1$. The Riemann Hypothesis conjectures that all non-trivial zeros lie on $\Re(s) = 1/2$. Functional equation: $\zeta(s) = 2^s \pi^{s-1} \sin(\pi s/2) \Gamma(1-s) \zeta(1-s)$.""",
    r"""Maxwell's equations in differential form: $\nabla \cdot \mathbf{E} = \rho/\varepsilon_0$, $\nabla \cdot \mathbf{B} = 0$, $\nabla \times \mathbf{E} = -\partial \mathbf{B}/\partial t$, $\nabla \times \mathbf{B} = \mu_0(\mathbf{J} + \varepsilon_0 \partial \mathbf{E}/\partial t)$.""",
    r"""For the Schrödinger equation $i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi$ with $\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r})$, separable solutions take the form $\psi(\mathbf{r}, t) = \phi(\mathbf{r})e^{-iEt/\hbar}$ where $\hat{H}\phi = E\phi$.""",
]

MIXED_PROMPTS = [
    "🎉🚀💻🔥📊📈🎯✨🌟💡🎨🔧⚙️🛠️🧪🔬🌍🌎🌏🌐🗺️🧭🚢✈️🚁🛸🛰️🛤️🚄🚅🚈🚉🛣️🛬🛫⛽🚦🚥🚧 Translate: Each emoji represents a concept; concatenate descriptions.",
    "Decode this base64 payload and explain what it does: aW1wb3J0IHRpa3Rva2VuOyBlbmMgPSB0aWt0b2tlbi5nZXRfZW5jb2RpbmcoJ2NsMTAwa19iYXNlJyk7IHRva2VucyA9IGVuYy5lbmNvZGUoIkhlbGxvLCB3b3JsZCEiKQ== Then suggest two alternative encodings.",
    """Parse this configuration: {"設定": {"予算": {"上限_uc": 540, "余裕率": 2.0, "プロバイダ": {"OpenAI": {"単価": 0.15}, "Anthropic": {"単価": 1.0, "_注釈": "クロード3.5以降"}, "Groq": null}, "_デバッグ": true}, "ログ": ["INFO: 初期化完了", "WARN: タイマー設定 ⏰", "ERROR: 認証失敗 🔒"]}}""",
    r"""The function `compute_budget_margin` returns $\mu = 2.0$ for Anthropic. In Rust: ```rust
let margin: f64 = match provider {
    Provider::Anthropic => 2.0_f64,  // 安全マージン
    Provider::OpenAI    => 1.4_f64,  // 较低 margin
    _ => 1.0_f64,
};
let predicted_uc = (byte_len as f64 * margin) as u64;
```
where $\mathrm{byte\_len}$ is measured by `.len()` on the serialised message body.""",
    "Fetch these endpoints in order: https://api.example.com/v1/budgets?owner=用户123&limit=10 https://docs.example.com/索引?lang=zh-CN https://github.com/sajjadanwar0/токен-бюджеты/issues?q=label%3Aбагов Then report HTTP status codes.",
    "🔐 H4sIAAAAAAAAA00OuwrCQBBE+xT5lP0LMXFNJTYWFmqsBC2tZGOuiSEPyG0Eg/679w0xQjKwwOzMzM6OkhKDLEgEbF54zw3O5dgxRSqYwlR+wfYbqGRYJjz0v9NEEEbgEYHaR3Vqs5iLM2lAGYAOdv7AABTPa/yvDBLHJEbUklEqWvNRQ4hwY7Bh== 🚀 Apply rot13 then base64-decode.",
    r"""Given: $f(n) = O(n \log n)$ where $\log = \log_2$. Implementation:
def msort(xs: list[int]) -> list[int]:
    if len(xs) <= 1: return xs
    m = len(xs) // 2
    L, R = msort(xs[:m]), msort(xs[m:])
    return merge(L, R)  # 마지막에 병합
Time: $T(n) = 2T(n/2) + O(n) = O(n \log n)$. 空间复杂度: $O(n)$.""",
    "Café vs Café (different normalizations). Hebrew shalom: שָׁלוֹם with nikud. Mathematical script: 𝒜 𝓁 𝓰 𝑒 𝓫 𝓻 𝒶. Emoji ZWJ sequences: 👨‍👩‍👧‍👦 (family), 🏳️‍🌈 (rainbow flag). RTL mark test: a‏b c.",
    """请实现一个 budget tracker in Rust。Requirements:
1. 使用 affine ownership (i.e., implement Drop, don't implement Clone).
2. `pub fn spend(self, amount: u64) -> Result<Self, BudgetError>` - 注意是 by-value self.
3. Add a #[must_use] attribute to ensure 编译器警告 if the result is dropped.
Example usage in main():
```rust
let b = Budget::new(1000);
let b = b.spend(300)?;  // 还剩 700
```
Return the full implementation 加上 unit tests.""",
    "aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa",
]


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic()

    corpus = []
    for prompt in CJK_PROMPTS:
        corpus.append(("CJK_heavy", prompt))
    for prompt in CODE_PROMPTS:
        corpus.append(("code_heavy", prompt))
    for prompt in MATH_PROMPTS:
        corpus.append(("math_heavy", prompt))
    for prompt in MIXED_PROMPTS:
        corpus.append(("mixed_edge", prompt))

    print("=" * 88)
    print("Independent Adversarial Hold-Out Validation for AnthropicEstimator")
    print("=" * 88)
    print(f"Model:    {MODEL}")
    print(f"Margin:   {MARGIN}x (predicted_tokens = byte_length * {MARGIN})")
    print(f"Corpus:   {len(corpus)} prompts across 4 adversarial categories")
    print(f"          (independently generated, no overlap with calibration audit)")
    print("=" * 88)

    rows = []
    cat_stats = {}

    for idx, (category, prompt) in enumerate(corpus, 1):
        bl = len(prompt.encode("utf-8"))
        predicted_tokens = bl * MARGIN
        try:
            resp = client.messages.count_tokens(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            actual_tokens = resp.input_tokens
        except Exception as e:
            print(f"  [{idx:02d}] {category:<14} FAILED: {e}")
            continue

        ratio = predicted_tokens / actual_tokens if actual_tokens > 0 else float("inf")
        a1_satisfied = predicted_tokens >= actual_tokens

        rows.append({
            "idx": idx,
            "category": category,
            "byte_length": bl,
            "predicted_tokens": predicted_tokens,
            "actual_tokens": actual_tokens,
            "ratio": ratio,
            "a1_satisfied": a1_satisfied,
        })

        c = cat_stats.setdefault(category, {"n": 0, "ok": 0, "ratios": []})
        c["n"] += 1
        if a1_satisfied:
            c["ok"] += 1
        c["ratios"].append(ratio)

        status = "OK " if a1_satisfied else "FAIL"
        print(f"  [{idx:02d}] {category:<14} bl={bl:>4}  pred={predicted_tokens:>6.0f}  "
              f"actual={actual_tokens:>4}  ratio={ratio:>5.2f}  {status}")

    print()
    print("=" * 88)
    print("Per-category summary")
    print("=" * 88)
    for cat in ["CJK_heavy", "code_heavy", "math_heavy", "mixed_edge"]:
        if cat not in cat_stats:
            continue
        c = cat_stats[cat]
        rs = c["ratios"]
        print(f"  {cat:<14} {c['ok']:>2}/{c['n']:>2} A1 satisfied   "
              f"ratio min={min(rs):.2f} mean={sum(rs) / len(rs):.2f} max={max(rs):.2f}")

    # Overall
    total_n = len(rows)
    total_ok = sum(1 for r in rows if r["a1_satisfied"])
    overall_ratios = [r["ratio"] for r in rows]
    overall_min = min(overall_ratios)
    overall_mean = sum(overall_ratios) / len(overall_ratios)
    overall_max = max(overall_ratios)

    if total_n > 0:
        p = total_ok / total_n
        z = 1.96
        denom = 1 + z * z / total_n
        center = (p + z * z / (2 * total_n)) / denom
        radius = (z / denom) * math.sqrt(p * (1 - p) / total_n + z * z / (4 * total_n * total_n))
        lo, hi = max(0.0, center - radius), min(1.0, center + radius)
    else:
        lo, hi = 0.0, 0.0

    print()
    print("=" * 88)
    print("Overall")
    print("=" * 88)
    print(f"  A1 satisfied:                {total_ok}/{total_n}")
    print(f"  Wilson 95% CI on rate:       [{lo:.3f}, {hi:.3f}]")
    print(f"  Margin ratios:               min={overall_min:.2f}  mean={overall_mean:.2f}  max={overall_max:.2f}")
    if total_ok < total_n:
        fails = [r for r in rows if not r["a1_satisfied"]]
        print(f"  FAILED prompts ({len(fails)}):")
        for r in fails:
            print(f"    [{r['idx']:02d}] {r['category']}: bl={r['byte_length']} "
                  f"pred={r['predicted_tokens']:.0f} actual={r['actual_tokens']} ratio={r['ratio']:.3f}")
    else:
        worst = min(rows, key=lambda r: r["ratio"])
        print(f"  Worst case (lowest ratio):   [{worst['idx']:02d}] {worst['category']} "
              f"bl={worst['byte_length']} pred={worst['predicted_tokens']:.0f} "
              f"actual={worst['actual_tokens']} ratio={worst['ratio']:.3f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "idx", "category", "byte_length", "predicted_tokens",
            "actual_tokens", "ratio", "a1_satisfied",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print()
    print(f"CSV written: {OUTPUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()