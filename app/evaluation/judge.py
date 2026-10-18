"""LLM-as-judge for answer faithfulness and relevance.

The judge model is configured separately (JUDGE_MODEL) from the generator (GROQ_MODEL) and should be
from a different model family where possible to limit self-enhancement bias. Judge scores are an
automated proxy, NOT human ground truth: they inherit the judge's own biases and noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.exceptions import GenerationError
from app.generation.answer_service import parse_model_json
from app.generation.groq_client import GroqClient

JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers. You are given a QUESTION,
the retrieved CONTEXT chunks, and an ANSWER. Judge ONLY from the context; ignore your own knowledge.

Score two criteria on an integer scale 1-5:
FAITHFULNESS (groundedness): are all factual claims in the ANSWER supported by the CONTEXT?
  5 = every claim is directly supported; 4 = supported with trivial paraphrase; 3 = mostly supported but one minor
  unsupported detail; 2 = a material claim is unsupported; 1 = mostly unsupported or contradicts the context.
RELEVANCE: does the ANSWER address the QUESTION asked?
  5 = fully and directly answers; 3 = partially answers or adds much irrelevant material; 1 = does not answer.
If the ANSWER correctly states that the context lacks the information, FAITHFULNESS is 5, and RELEVANCE is 5
only if the context really does lack it, otherwise 2.

Respond with ONE JSON object only:
{"faithfulness": <1-5>, "relevance": <1-5>, "supported_claims": ["..."], "unsupported_claims": ["..."], "rationale": "<one or two sentences>"}"""


@dataclass
class JudgeResult:
    available: bool
    faithfulness: float | None = None      # normalised to 0..1 ((score-1)/4)
    relevance: float | None = None
    raw_faithfulness: int | None = None
    raw_relevance: int | None = None
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    rationale: str = ""
    judge_model: str = ""
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _norm(v) -> tuple[int | None, float | None]:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None, None
    if not 1 <= i <= 5:
        return None, None
    return i, (i - 1) / 4


class LLMJudge:
    def __init__(self, settings: Settings, client: GroqClient | None = None) -> None:
        self.settings = settings
        self.client = client or GroqClient(settings)
        self.model = settings.judge_model

    @property
    def same_as_generator(self) -> bool:
        return self.model == self.settings.groq_model

    def judge(self, question: str, answer: str, context_chunks: list[str]) -> JudgeResult:
        ctx = "\n".join(f'<chunk ref="C{i}">\n{t}\n</chunk>' for i, t in enumerate(context_chunks, start=1))
        user = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nReturn only the JSON object."
        try:
            resp = self.client.complete([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                                        model=self.model, max_tokens=400, temperature=0.0)
        except GenerationError as exc:
            return JudgeResult(available=False, judge_model=self.model, error=f"{exc.category}: {exc.message}")
        obj = parse_model_json(resp.text)
        if obj is None:
            return JudgeResult(available=False, judge_model=self.model, error="malformed judge output")
        rf, f = _norm(obj.get("faithfulness"))
        rr, r = _norm(obj.get("relevance"))
        if f is None or r is None:
            return JudgeResult(available=False, judge_model=self.model, error="judge scores missing or out of range")
        return JudgeResult(True, f, r, rf, rr, [str(x) for x in obj.get("supported_claims", [])][:10],
                           [str(x) for x in obj.get("unsupported_claims", [])][:10],
                           str(obj.get("rationale", ""))[:500], self.model, None,
                           resp.prompt_tokens, resp.completion_tokens)
