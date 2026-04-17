"""
LangGraph Formal Orchestrator for DraftClear Pipeline
Implements cyclical workflow with state management
"""
import logging
from typing import Literal
from utils.drawing_state import DrawingState

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, will use fallback")

# Import all agents
from agents.perception import PerceptionAgent
from agents.masking import MaskingAgent
from agents.spatial_resolution import SpatialResolutionAgent
from agents.healing import HealingAgent
from agents.supervisor import SupervisorAgent


class LangGraphOrchestrator:
    """LangGraph-based orchestration of DraftClear agents"""

    def __init__(self):
        """Initialize orchestrator with all agents"""
        logger.info("Initializing LangGraph Orchestrator")
        self.perception = PerceptionAgent()
        self.masking = MaskingAgent()
        self.spatial_resolution = SpatialResolutionAgent()
        self.healing = HealingAgent()
        self.supervisor = SupervisorAgent()

        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_langgraph()
            logger.info("LangGraph workflow created")
        else:
            logger.warning("LangGraph not available, using fallback orchestration")

    def _build_langgraph(self):
        """Build LangGraph state machine"""
        workflow = StateGraph(DrawingState)

        # Add nodes for each agent
        workflow.add_node("perception", self._node_perception)
        workflow.add_node("masking", self._node_masking)
        workflow.add_node("spatial_resolution", self._node_spatial_resolution)
        workflow.add_node("healing", self._node_healing)
        workflow.add_node("supervisor", self._node_supervisor)

        # Define edge routing
        workflow.set_entry_point("perception")

        # Perception -> Masking (always)
        workflow.add_edge("perception", "masking")

        # Masking -> Spatial Resolution (always)
        workflow.add_edge("masking", "spatial_resolution")

        # Spatial Resolution -> Healing (always)
        workflow.add_edge("spatial_resolution", "healing")

        # Healing -> Supervisor (always)
        workflow.add_edge("healing", "supervisor")

        # Supervisor -> routing decision
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_route,
            {
                "loop": "spatial_resolution",  # Loop back to spatial resolution
                "compile": END,  # End pipeline
            }
        )

        return workflow.compile()

    def _node_perception(self, state: DrawingState) -> DrawingState:
        """Perception node"""
        return self.perception.run(state)

    def _node_masking(self, state: DrawingState) -> DrawingState:
        """Masking node"""
        return self.masking.run(state)

    def _node_spatial_resolution(self, state: DrawingState) -> DrawingState:
        """Spatial resolution node"""
        return self.spatial_resolution.run(state)

    def _node_healing(self, state: DrawingState) -> DrawingState:
        """Healing node"""
        return self.healing.run(state)

    def _node_supervisor(self, state: DrawingState) -> DrawingState:
        """Supervisor node"""
        return self.supervisor.run(state)

    def _supervisor_route(self, state: DrawingState) -> Literal["loop", "compile"]:
        """Route decision based on supervisor"""
        if state.supervisor_decision == "compile":
            return "compile"
        else:
            return "loop"

    def run(self, state: DrawingState) -> DrawingState:
        """Execute the orchestrated workflow"""
        logger.info("Starting LangGraph workflow execution")

        if LANGGRAPH_AVAILABLE:
            try:
                # Run graph
                result = self.graph.invoke(state)

                # Extract DrawingState from result if it's a dict
                if isinstance(result, dict):
                    # LangGraph returns the final state from the graph
                    # The result should contain the DrawingState or its attributes
                    if isinstance(result, DrawingState):
                        return result
                    else:
                        # If it's a dict, try to convert back to DrawingState
                        logger.warning("LangGraph returned dict instead of DrawingState, using fallback")
                        return self._run_simple_orchestration(state)
                else:
                    return result
            except Exception as e:
                logger.error(f"LangGraph execution error: {e}")
                logger.info("Falling back to simple orchestration")
                return self._run_simple_orchestration(state)
        else:
            return self._run_simple_orchestration(state)

    def _run_simple_orchestration(self, state: DrawingState) -> DrawingState:
        """Fallback simple orchestration without LangGraph"""
        logger.info("Running simple orchestration")

        current_state = state
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Orchestration Iteration {iteration}")
            logger.info(f"{'='*60}")

            # Execute pipeline
            current_state = self.perception.run(current_state)
            if not current_state.text_boxes:
                logger.warning("No text detected, ending pipeline")
                break

            current_state = self.masking.run(current_state)
            current_state = self.spatial_resolution.run(current_state)
            current_state = self.healing.run(current_state)
            current_state = self.supervisor.run(current_state)

            # Check supervisor decision
            if current_state.supervisor_decision == "compile":
                logger.info("Supervisor approved for compilation")
                break

        return current_state
