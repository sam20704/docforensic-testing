import asyncio
import re
import traceback
from datetime import datetime, timezone

from digitrase_vision.business_service_base import BusinessServiceBase
from digitrase_vision.config.app_setting import VisionAppSettings
from digitrase_vision.data_access.models import MetaVisionModel, ForensicDocumentModel, ForensicResponseModel, \
    FontPageModel, QrPageModel, VisionStatus, ResultFileModel
from digitrase_vision.modules.artefact_module.artefact_ai_model_handler import ArtefactAIModelsHandler
from digitrase_vision.modules.template_matching.Image_Embedding_Model.extract_features import ImageEmbeddingHandler
from digitrase_vision.modules.ai_detector.ai_detector_model_handler import AIDetectorHandler
from digitrase_vision.modules.ai_detector.inference import AIDetectorInference
from digitrase_vision.modules.template_matching.search_template import SearchTemplate
from digitrase_vision.modules.font_module.font_analysis import FontAnalysis
from digitrase_vision.modules.meta_module.meta_validation import MetaValidation
from digitrase_vision.modules.qr_module.qr_validation import QrValidation
from digitrase_vision.modules.artefact_module.compression_detection import CompressionDetection
from digitrase_shared.abstractions.document_store import DocumentStore
from digitrase_shared.abstractions.text_extractor_service import TextExtractorService
from digitrase_shared.common.digitrase_error import DigitraseError, ErrorStatus
from digitrase_shared.data_access.mongo_data_access import MongoDataAccess
from digitrase_vision.config.business_settings import BusinessSettings
from digitrase_vision.modules.draw_results.draw_bbox import DrawResults
from digitrase_shared.vectordb.qdrantdb import QdrantDB
from digitrase_vision.services.reasoning_service import ReasoningService


class MetaVisionService(BusinessServiceBase):

    def __init__(self, document_store: DocumentStore, data_access: MongoDataAccess[MetaVisionModel],
                 app_settings: VisionAppSettings, text_extractor: TextExtractorService,
                 
                 business_settings: BusinessSettings, artefact_ai_models_handler: ArtefactAIModelsHandler,
                 aidetector_models_handler: AIDetectorHandler,
                 template_image_models_handler: ImageEmbeddingHandler,
                 qdrantdb_client: QdrantDB,
                 logger):
        """
        Initialize the QueueProcessor with a DI container.

        """
        super().__init__(document_store, data_access, app_settings, business_settings, logger)
        self.text_extractor = text_extractor
        self.artefact_ai_models_handler = artefact_ai_models_handler
        self.aidetector_models_handler = aidetector_models_handler
        self.template_image_models_handler = template_image_models_handler
        self.qdrantdb_client = qdrantdb_client
        self.reasoning_service = ReasoningService(logger)

    async def process(self, incoming_data):

        try:
            self.logger.info(f"received incoming msg {incoming_data}")

            model = await self.data_access.read_by_ingestion_id(incoming_data['file_store_id'])
            if model is None:
                model = MetaVisionModel(ingestion_id=incoming_data['file_store_id'])
            else:
                model.updated_date = datetime.now(timezone.utc)

            # process started time
            model.processing_started_at = datetime.now(timezone.utc)

            await self.data_access.upsert(model)
            # incoming_data = {"original_name": "", "file_store_id": "", "converted_images": {}}
            original_files = [incoming_data]



            semaphore = asyncio.Semaphore(
                self.app_settings.module_settings.parallel_executions)  # Default Limit to 2 concurrent tasks

            tasks = []
            for original_file in original_files:
                async with semaphore:
                    # Offload CPU-intensive task to a thread
                    tasks.append(
                        await asyncio.to_thread(self.process_file, original_file["original_name"], original_file["file_store_id"],
                           original_file['converted_images']))

            results = await asyncio.gather(*tasks)
            model.documents = results
            model.status = VisionStatus.Completed
            # process completed time
            model.processing_completed_at = datetime.now(timezone.utc)
            await self.data_access.update_entry(model)
            self.logger.info(f"Message: {incoming_data} processed.")
        except Exception as e:
            self.logger.error(f"Error processing message: {incoming_data}. Error: {e}")
            await self.data_access.update_data({"ingestion_id": incoming_data['ingestion_id']},
                                               {"status": VisionStatus.Failed})
            raise e

    async def process_file(self, original_name, file_store_id,
                           file_store_ids) -> ForensicDocumentModel:

        try:
            # sort file_store_ids based on keys which in int
            file_store_ids = dict(sorted(file_store_ids.items()))
            file_store_ids = list(file_store_ids.values())

            module_setting = {"modules": []}
            for job in self.business_settings.job_list:
                if job.job_name == "meta_vision":
                    for e_mod in job.job_params.modules:
                        if re.compile(str(e_mod.module_config.file_format), re.IGNORECASE).search(
                                original_name):
                            module_setting["modules"].append(e_mod.module_name)

            if not module_setting:
                raise DigitraseError(status=ErrorStatus.BadRequest,
                                     message=f"File format not supported: {original_name}")


            file_content = self.document_store.download_file(file_store_id)

            # list of images in pdf
            related_file_contents = self.document_store.download_files(file_store_ids)

            document_result: ForensicDocumentModel = ForensicDocumentModel(file_store_id=file_store_id,
                                                                           original_file_name=original_name)

            draw_result = DrawResults(document_store=self.document_store, logger=self.logger)

            for module in module_setting.get("modules"):
                if module == "font_module":
                    font_module = FontAnalysis(document_store=self.document_store,
                                               business_settings=self.business_settings, logger=self.logger)
                    font_analysis_result = await font_module.process(file_content, [file_content for _, file_content in
                                                                                    related_file_contents],
                                                                     original_name)
                    font_analysis = ForensicResponseModel[FontPageModel](pages=font_analysis_result.pages,
                                                                         score=font_analysis_result.score,
                                                                         reasons=font_analysis_result.reasons)

                    document_result.font_analysis = font_analysis
                elif module == "artefact_module":

                    artefact_module = CompressionDetection(document_store=self.document_store,
                                                           business_settings=self.business_settings,
                                                           model=self.artefact_ai_models_handler.model,
                                                           logger=self.logger)

                    artefact_analysis_result = await artefact_module.process(file_content,
                                                                             [file_content for _, file_content in
                                                                              related_file_contents],
                                                                             original_name)
                    document_result.artefact_analysis = artefact_analysis_result

                elif module == "aidetector_module":

                    aidetector_module = AIDetectorInference(document_store=self.document_store,
                                                           business_settings=self.business_settings,
                                                           model=self.aidetector_models_handler.model,
                                                           logger=self.logger)

                    aidetector_analysis_result = await aidetector_module.process(file_content,
                                                                             [file_content for _, file_content in
                                                                              related_file_contents],
                                                                             original_name)
                    document_result.aidetector_analysis = aidetector_analysis_result

                elif module == "template_module":

                    template_module = SearchTemplate(document_store=self.document_store,
                                                           business_settings=self.business_settings,
                                                           image_model=self.template_image_models_handler,
                                                           qdrantdb_client=self.qdrantdb_client,
                                                           logger=self.logger)

                    template_analysis_result = await template_module.process(file_content,
                                                                             [file_content for _, file_content in
                                                                              related_file_contents],
                                                                             original_name)
                    document_result.template_analysis = template_analysis_result

                elif module == "meta_module":
                    meta_module = MetaValidation(document_store=self.document_store, text_extractor=self.text_extractor,
                                                 business_settings=self.business_settings, logger=self.logger)

                    meta_analysis_result, image_analysis_result = await meta_module.process(file_content, [file_content for _, file_content in
                                                                                    related_file_contents],
                                                                     original_name)
                    document_result.meta_analysis = meta_analysis_result
                    document_result.image_analysis = image_analysis_result

                elif module == "qr_module":
                    # call qr module
                    qr_module = QrValidation(document_store=self.document_store, text_extractor=self.text_extractor,
                                             business_settings=self.business_settings, logger=self.logger)
                    qr_analysis_result = await qr_module.process(file_content, [file_content for _, file_content in
                                                                                related_file_contents],
                                                                 original_name)
                    qr_analysis = ForensicResponseModel[QrPageModel](pages=qr_analysis_result.pages,score=qr_analysis_result.score, reasons=qr_analysis_result.reasons)

                    document_result.qr_analysis = qr_analysis

            # draw results
            if original_name.lower().endswith(".pdf"):
                images = [file_content for _, file_content in related_file_contents]

            elif original_name.lower().endswith(".png") or original_name.lower().endswith(
                    ".jpg") or original_name.lower().endswith(".jpeg"):
                images = [file_content]

            else:
                images = []

            page_wise_result_file_id = await draw_result.draw_all(document_result=document_result, images=images)
            if page_wise_result_file_id is not None:
                document_result.combined_result_files = [
                    ResultFileModel(file_store_id=file_id, page_number=page_num)
                    for page_num, file_id in page_wise_result_file_id.items()
                ]

            document_result = await draw_result.combine_score_reason(document_result)
            # Generate forensic reasoning
            print("Generating LLM reasoning...")
            document_result.reasoning = self.reasoning_service.generate_reasoning(document_result)
            self.logger.info(f"Document {document_result} processed.")

            return document_result
        except Exception as e:
            # debug purpose
            print(traceback.print_exc())
            raise f"Exception raised at metavisionservice process file {e}"

