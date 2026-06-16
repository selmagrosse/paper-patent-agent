"""ValidatorAgent — judges whether a search candidate genuinely satisfies a user request."""

import json

from dotenv import load_dotenv
from openai import OpenAI

from base_agent import BaseAgent

load_dotenv()

MODEL = "gpt-4o-mini"

VALIDATE_PROMPT = """\
A user made the following request: "{user_request}"

A search returned this candidate result:
Title: {title}
Abstract: {abstract}

Judge whether this candidate genuinely satisfies the user's request.

Return ONLY a valid JSON object with exactly this structure:
{{
    "valid": true or false,
    "confidence": <float between 0 and 1>,
    "reasoning": "one sentence explaining the judgment"
}}"""


class ValidatorAgent(BaseAgent):
    """Agent that validates whether a candidate result matches a user request.

    Used via composition by other agents (e.g. PaperPatentAgent) to
    double-check search results before they're presented or downloaded.
    """

    def __init__(self) -> None:
        super().__init__(model_name=MODEL)
        self._client = OpenAI()

    def validate(self, user_request: str, candidate: dict) -> dict:
        """Judge whether *candidate* genuinely satisfies *user_request*.

        Makes a single LLM call with the user's request and the candidate's
        title and abstract (truncated to 300 characters), and asks the model
        to return a structured validity judgment.

        Args:
            user_request: The original natural-language request from the user.
            candidate: A result dict containing at least ``title`` and
                ``abstract`` keys.

        Returns:
            Dict with keys ``valid`` (bool), ``confidence`` (float 0-1), and
            ``reasoning`` (str). On parse failure, returns a permissive default:
            ``{"valid": True, "confidence": 0.5, "reasoning": "validation parsing failed"}``.
        """
        prompt = VALIDATE_PROMPT.format(
            user_request=user_request,
            title=candidate.get("title", ""),
            abstract=candidate.get("abstract", "")[:300],
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "valid": True,
                "confidence": 0.5,
                "reasoning": "validation parsing failed",
            }

    def run(self, user_request: str) -> str:
        """Not used directly — validation is invoked via :meth:`validate`.

        Args:
            user_request: Natural-language request from the user.

        Returns:
            A placeholder string, since this agent is driven by direct
            method calls rather than the standard run loop.
        """
        return self.llm_call(user_request)
