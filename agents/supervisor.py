import logging
import requests
import json
from config.settings import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, MAX_ITERATIONS
from utils.drawing_state import DrawingState

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """LLM-based supervisor using Ollama for intelligent decision making"""

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        """
        Initialize supervisor with Ollama connection

        Args:
            ollama_url: URL to Ollama server
            model: Model name to use
        """
        self.ollama_url = ollama_url
        self.model = model
        self.is_available = self._check_ollama_availability()

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Using fallback logic.")
            return False

    def call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with prompt

        Args:
            prompt: Input prompt for LLM

        Returns:
            LLM response text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temperature for deterministic output
                    "top_p": 0.8,
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""

    def parse_decision(self, response: str) -> str:
        """
        Parse LLM response to extract decision

        Args:
            response: LLM response text

        Returns:
            "continue" or "compile"
        """
        response_lower = response.lower()

        if "compile" in response_lower or "done" in response_lower or "complete" in response_lower:
            return "compile"
        elif "continue" in response_lower or "loop" in response_lower or "repeat" in response_lower:
            return "continue"
        else:
            # Default to continue if ambiguous
            return "continue"

    def run(self, state: DrawingState) -> DrawingState:
        """
        Execute supervisor agent to make routing decision

        Args:
            state: Current DrawingState

        Returns:
            Updated DrawingState with supervisor decision
        """
        logger.info(f"[Iteration {state.iteration}] Running Supervisor Agent")

        new_state = state.copy()

        # Check iteration limit
        if state.iteration >= MAX_ITERATIONS:
            logger.warning(f"Maximum iterations ({MAX_ITERATIONS}) reached, forcing compilation")
            new_state.supervisor_decision = "compile"
            new_state.supervisor_reasoning = "Max iterations reached"
            return new_state

        # If Ollama is available, use it for intelligent decision making
        if self.is_available:
            logger.info("Using Ollama for intelligent supervision")

            prompt = f"""You are a CAD drawing repair supervisor. Analyze this situation and decide whether to continue repositioning text labels or if the drawing is ready for compilation.

Current State:
- Iteration: {state.iteration}
- Collision count: {state.collision_count}
- Total text boxes: {len(state.text_boxes)}

Decision: If collision_count is 0, respond "COMPILE". Otherwise respond "CONTINUE".
Be brief."""

            response = self.call_ollama(prompt)
            logger.info(f"Ollama response: {response}")

            decision = self.parse_decision(response)
            new_state.supervisor_decision = decision
            new_state.supervisor_reasoning = response[:100]  # Store first 100 chars

        else:
            # Fallback: simple logic without LLM
            logger.info("Ollama unavailable, using fallback logic")

            if state.collision_count == 0:
                new_state.supervisor_decision = "compile"
                new_state.supervisor_reasoning = "No collisions detected"
            else:
                new_state.supervisor_decision = "continue"
                new_state.supervisor_reasoning = f"{state.collision_count} collisions remaining"

        logger.info(f"Supervisor decision: {new_state.supervisor_decision}")

        return new_state
