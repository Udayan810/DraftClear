"""
Main DraftClear orchestrator using LangGraph
Coordinates all agents in the cyclical AI pipeline
"""
import logging
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import TypedDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agents
from agents.perception import PerceptionAgent
from agents.masking import MaskingAgent
from agents.spatial_resolution import SpatialResolutionAgent
from agents.healing import HealingAgent
from agents.supervisor import SupervisorAgent
from utils.drawing_state import DrawingState
from config.settings import OUTPUTS_DIR

# Try to import LangGraph, fallback to manual orchestration if not available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangGraph not available, using simple orchestration")
    LANGGRAPH_AVAILABLE = False


class DraftClearPipeline:
    """Main pipeline orchestrator for DraftClear"""

    def __init__(self):
        """Initialize all agents"""
        logger.info("Initializing DraftClear Pipeline...")
        self.perception = PerceptionAgent()
        self.masking = MaskingAgent()
        self.spatial_resolution = SpatialResolutionAgent()
        self.healing = HealingAgent()
        self.supervisor = SupervisorAgent()
        logger.info("All agents initialized successfully")

    def process_image(self, image_path: str, output_prefix: str = "output") -> DrawingState:
        """
        Process a single image through the complete pipeline

        Args:
            image_path: Path to input image
            output_prefix: Prefix for output files

        Returns:
            Final DrawingState with results
        """
        logger.info(f"Processing image: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        logger.info(f"Image loaded: shape={image.shape}")

        # Initialize state
        state = DrawingState(original_image=image)

        # Run pipeline loop
        max_loops = 5
        loop_count = 0

        while loop_count < max_loops:
            loop_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Pipeline Loop {loop_count}")
            logger.info(f"{'='*60}")

            # Step 1: Perception Agent (detect text)
            state = self.perception.run(state)
            if not state.text_boxes:
                logger.warning("No text detected!")
                break

            # Step 2: Masking Agent (remove overlaps)
            state = self.masking.run(state)

            # Step 3: Spatial Resolution Agent (find safe positions)
            state = self.spatial_resolution.run(state)

            # Step 4: Healing Agent (repair geometry)
            state = self.healing.run(state)

            # Step 5: Supervisor Agent (make decision)
            state = self.supervisor.run(state)

            # Check supervisor decision
            if state.supervisor_decision == "compile":
                logger.info("Supervisor approved for compilation")
                break
            else:
                logger.info(f"Supervisor requesting loop back (collision_count={state.collision_count})")

        # Save outputs
        self.save_results(state, output_prefix)

        return state

    def save_results(self, state: DrawingState, output_prefix: str):
        """Save pipeline results to disk"""
        logger.info(f"Saving results with prefix: {output_prefix}")

        try:
            # Save original
            if state.original_image is not None:
                path = OUTPUTS_DIR / f"{output_prefix}_00_original.png"
                cv2.imwrite(str(path), state.original_image)
                logger.info(f"Saved: {path}")

            # Save damaged geometry
            if state.damaged_geometry is not None:
                path = OUTPUTS_DIR / f"{output_prefix}_01_damaged.png"
                cv2.imwrite(str(path), state.damaged_geometry)
                logger.info(f"Saved: {path}")

            # Save healed geometry
            if state.healed_geometry is not None:
                path = OUTPUTS_DIR / f"{output_prefix}_02_healed.png"
                cv2.imwrite(str(path), state.healed_geometry)
                logger.info(f"Saved: {path}")

            # Save report
            report_path = OUTPUTS_DIR / f"{output_prefix}_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"DraftClear Processing Report\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Iterations: {state.iteration}\n")
                f.write(f"Text Labels Detected: {len(state.text_boxes)}\n")
                f.write(f"Final Collision Count: {state.collision_count}\n")
                f.write(f"Supervisor Decision: {state.supervisor_decision}\n")
                f.write(f"Supervisor Reasoning: {state.supervisor_reasoning}\n")
                f.write(f"\nText Box Coordinates:\n")
                for i, box in enumerate(state.new_coordinates):
                    f.write(f"  Box {i}: ({box.x:.1f}, {box.y:.1f}) - {box.w:.1f}x{box.h:.1f}\n")

            logger.info(f"Saved: {report_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DraftClear - CAD Label Placement Problem Solver")
    parser.add_argument("--input", type=str, required=False, help="Input image path")
    parser.add_argument("--test", action="store_true", help="Run with test image")
    parser.add_argument("--generate-test-data", action="store_true", help="Generate test data")

    args = parser.parse_args()

    # Generate test data if requested
    if args.generate_test_data:
        logger.info("Generating test data...")
        import generate_test_data
        generate_test_data.create_simple_blueprint("data/test_inputs/simple_blueprint.png")
        generate_test_data.create_overlap_example("data/test_inputs/overlap_example.png")

    # Run pipeline
    pipeline = DraftClearPipeline()

    if args.test or args.input is None:
        # Run test
        test_image = Path("data/test_inputs/simple_blueprint.png")
        if not test_image.exists():
            logger.error(f"Test image not found: {test_image}")
            logger.info("Run with --generate-test-data first")
            return

        logger.info("Running test pipeline...")
        state = pipeline.process_image(str(test_image), "test_simple")

    elif args.input:
        # Run on custom image
        state = pipeline.process_image(args.input, "custom")

    # Print summary
    if state:
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Iterations: {state.iteration}")
        logger.info(f"Text labels: {len(state.text_boxes)}")
        logger.info(f"Final collisions: {state.collision_count}")
        logger.info(f"Decision: {state.supervisor_decision}")


if __name__ == "__main__":
    main()
