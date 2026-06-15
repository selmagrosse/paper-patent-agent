"""Base class for agents in the paper-patent pipeline."""


class BaseAgent:
    """Minimal agent base providing memory, a tool registry, and LLM access.

    Attributes:
        model_name: OpenAI model identifier used for LLM calls.
        max_memory_size: Maximum number of turns kept in memory. Oldest
            entries are evicted when the limit is exceeded.
        memory: Ordered list of conversation turns, each a dict with
            ``role`` (``"user"`` or ``"agent"``) and ``content`` keys.
        tools: Registry mapping tool name strings to callables.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_memory_size: int = 20,
    ) -> None:
        self.model_name = model_name
        self.max_memory_size = max_memory_size
        self.memory: list[dict] = []
        self.tools: dict[str, callable] = {}

    def add_to_memory(self, role: str, content: str) -> None:
        """Append a turn to memory, evicting the oldest entry if over the limit.

        Args:
            role: Either ``"user"`` or ``"agent"``.
            content: The message text.
        """
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def register_tool(self, name: str, fn: callable) -> None:
        """Register a callable under *name* for dispatch by :meth:`execute_tool`.

        Args:
            name: Tool identifier string.
            fn: The callable that implements the tool.
        """
        self.tools[name] = fn

    def execute_tool(self, name: str, **kwargs) -> any:
        """Look up and call a registered tool, catching any runtime errors.

        Args:
            name: Tool identifier previously registered via :meth:`register_tool`.
            **kwargs: Arguments forwarded to the tool callable.

        Returns:
            The tool's return value, or an error dict if the tool raises.
        """
        if name not in self.tools:
            return {"error": f"Unknown tool: {name!r}"}
        try:
            return self.tools[name](**kwargs)
        except Exception as exc:
            return {"error": f"Tool {name!r} failed: {exc}"}

    def llm_call(self, prompt: str) -> str:
        """Stub for a raw LLM completion — subclasses may override.

        Returns a formatted placeholder so the base class is usable without
        a live API connection during testing.

        Args:
            prompt: The prompt string to send to the model.

        Returns:
            A string indicating the model and prompt length.
        """
        return f"[{self.model_name}] received prompt ({len(prompt)} chars)"

    def run(self, user_request: str) -> str:
        """Process a user request end-to-end and return the agent's response.

        Subclasses should override this method with their full pipeline logic.

        Args:
            user_request: Natural-language request from the user.

        Returns:
            A string response to be shown to the user.
        """
        return self.llm_call(user_request)
