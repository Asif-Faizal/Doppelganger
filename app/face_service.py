import numpy as np
import cv2
import face_recognition
from PIL import Image
import io
from typing import Optional, List, Tuple
import logging
from sklearn.preprocessing import normalize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceNetService:
    def __init__(self):
        """Initialize face recognition using dlib's face_recognition library"""
        logger.info("Initializing face_recognition library...")
        
        # No initialization needed for face_recognition library
        # It uses HOG + Linear SVM for face detection and deep learning for face encoding
        
        logger.info("Face recognition service initialized successfully")
    

    
    def extract_face_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract facial embedding from image bytes using face_recognition library
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            128-dimensional face embedding as numpy array, or None if no face detected
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to numpy array (RGB format)
            img_array = np.array(image)
            
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(img_array, model="hog")
            
            if not face_locations:
                logger.warning("No face detected in the image")
                return None
            
            # Use the first (largest) face found
            face_location = face_locations[0]
            top, right, bottom, left = face_location
            
            logger.info(f"Face detected at location: top={top}, right={right}, bottom={bottom}, left={left}")
            
            # Extract face encodings (128-dimensional embedding)
            face_encodings = face_recognition.face_encodings(img_array, [face_location])
            
            if not face_encodings:
                logger.warning("Could not generate face encoding")
                return None
            
            embedding = face_encodings[0]
            
            logger.info(f"Successfully extracted embedding with shape: {embedding.shape}")
            logger.info(f"Embedding sample values: {embedding[:5]}")  # Log first 5 values for debugging
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings using face_recognition's distance metric
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (higher means more similar, range: 0 to 1)
        """
        # Calculate Euclidean distance (face_recognition uses this)
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity score (0-1 range, higher = more similar)
        # face_recognition typically uses 0.6 as threshold, so we scale accordingly
        similarity = max(0.0, 1.0 - (distance / 0.6))
        
        return float(similarity)
    
    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two face embeddings (face_recognition standard)
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Euclidean distance (lower means more similar)
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    def find_matches(self, query_embedding: np.ndarray, stored_embeddings: List[Tuple[str, np.ndarray]], 
                    threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find matching faces using face_recognition's distance metric
        
        Args:
            query_embedding: The embedding to search for
            stored_embeddings: List of (image_path, embedding) tuples
            threshold: Maximum distance for a match (default: 0.6, standard for face_recognition)
            
        Returns:
            List of (image_path, distance) tuples for matches, sorted by distance (lowest first)
        """
        matches = []
        
        for image_path, stored_embedding in stored_embeddings:
            distance = self.calculate_euclidean_distance(query_embedding, stored_embedding)
            
            # Use distance threshold (lower distance = better match)
            if distance <= threshold:
                matches.append((image_path, distance))
        
        # Sort by distance (lowest first = best matches first)
        matches.sort(key=lambda x: x[1])
        
        logger.info(f"Found {len(matches)} matches within distance threshold {threshold}")
        return matches

# Global instance
face_service = FaceNetService() 