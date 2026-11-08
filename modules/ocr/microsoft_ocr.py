import numpy as np
import imkit as imk

from .base import OCREngine
from .evidence import lowest_reported_score, make_ocr_evidence
from ..utils.textblock import TextBlock
from ..utils.textblock import lists_to_blk_list


class MicrosoftOCR(OCREngine):
    """OCR engine using Microsoft Azure Computer Vision API."""
    
    def __init__(self):
        self.client = None
        self.api_key = None
        self.endpoint = None
        
    def initialize(self, api_key: str, endpoint: str) -> None:
        """
        Initialize the Microsoft OCR with API key and endpoint.
        
        Args:
            api_key: Microsoft Azure API key
            endpoint: Microsoft Azure endpoint URL
            **kwargs: Additional parameters (ignored)
        """

        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.core.credentials import AzureKeyCredential

        self.api_key = api_key
        self.endpoint = endpoint
        self.client = ImageAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:

        from azure.ai.vision.imageanalysis.models import VisualFeatures

        texts_bboxes = []
        texts_string = []
        line_scores = []
        
        image_buffer = imk.encode_image(img, 'jpg')
        result = self.client.analyze(
            image_data=image_buffer, 
            visual_features=[VisualFeatures.READ]
        )
        
        if result.read and result.read.blocks:
            for line in result.read.blocks[0].lines:
                vertices = line.bounding_polygon
                
                # Ensure all vertices have both 'x' and 'y' coordinates
                if all('x' in vertex and 'y' in vertex for vertex in vertices):
                    x1 = vertices[0]['x']
                    y1 = vertices[0]['y']
                    x2 = vertices[2]['x']
                    y2 = vertices[2]['y']
                    
                    texts_bboxes.append((x1, y1, x2, y2))
                    texts_string.append(line.text)
                    words = getattr(line, "words", None) or []
                    line_scores.append(lowest_reported_score(getattr(word, "confidence", None) for word in words))
            
        def record_reading(block, indices):
            model = getattr(result, "model_version", None) or "Read"
            block.ocr_evidence = make_ocr_evidence(block.text, "Microsoft OCR", model,
                                                   line_scores=[line_scores[i] for i in indices])

        return lists_to_blk_list(blk_list, texts_bboxes, texts_string, on_text_assigned=record_reading)
