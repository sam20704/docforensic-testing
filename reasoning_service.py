# services/reasoning_service.py

from digitrase_vision.services.reasoning.reasoning_pipeline import run_reasoning_pipeline


class ReasoningService:

    def __init__(self, logger):
        self.logger = logger

    def generate_reasoning(self, document_result):

        try:

            reasoning = run_reasoning_pipeline(document_result)

            return reasoning

        except Exception as e:

            self.logger.error(f"LLM reasoning failed: {e}")

            return None
