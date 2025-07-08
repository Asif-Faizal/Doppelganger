from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
import aiofiles
from datetime import datetime
import logging

from .database import get_db, FaceEmbedding
from .face_service import face_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="High-accuracy face recognition API using FaceNet",
    version="1.0.0"
)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to storage directory"""
    # Create unique filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join("storage", "images", unique_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Face Recognition API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "face_service": "ready"
    }

@app.post("/upload-images")
async def upload_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload 1-2 images and extract facial embeddings using FaceNet
    
    Returns:
        JSON response with uploaded file information and embedding extraction results
    """
    if len(files) == 0 or len(files) > 2:
        raise HTTPException(
            status_code=400, 
            detail="Please upload 1 or 2 images"
        )
    
    results = []
    
    for file in files:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        try:
            # Read file content
            file_content = await file.read()
            
            # Extract face embedding
            embedding = face_service.extract_face_embedding(file_content)
            
            if embedding is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "No face detected in image"
                })
                continue
            
            # Save file
            await file.seek(0)  # Reset file pointer
            file_path = await save_uploaded_file(file)
            
            # Store in database
            db_embedding = FaceEmbedding(image_path=file_path)
            db_embedding.set_embedding(embedding)
            
            db.add(db_embedding)
            db.commit()
            db.refresh(db_embedding)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "image_path": file_path,
                "embedding_id": db_embedding.id,
                "embedding_shape": embedding.shape,
                "created_at": db_embedding.created_at.isoformat()
            })
            
            logger.info(f"Successfully processed {file.filename}")
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Processing failed: {str(e)}"
            })
    
    return {
        "message": f"Processed {len(files)} images",
        "results": results
    }

@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    threshold: float = Form(0.6),
    max_results: int = Form(10),
    db: Session = Depends(get_db)
):
    """
    Search for matching faces using Euclidean distance comparison (face_recognition standard)
    
    Args:
        file: Image file to search for
        threshold: Maximum distance for a match (default: 0.6, lower = more strict)
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        JSON response with matching images and their distance scores
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Extract face embedding from uploaded image
        file_content = await file.read()
        query_embedding = face_service.extract_face_embedding(file_content)
        
        if query_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the uploaded image"
            )
        
        # Get all stored embeddings
        stored_embeddings_db = db.query(FaceEmbedding).all()
        
        if not stored_embeddings_db:
            return {
                "message": "No faces in database to search against",
                "matches": []
            }
        
        # Convert to format needed for matching
        stored_embeddings = [
            (embedding_db.image_path, embedding_db.get_embedding())
            for embedding_db in stored_embeddings_db
        ]
        
        # Find matches
        matches = face_service.find_matches(
            query_embedding, 
            stored_embeddings, 
            threshold=threshold
        )
        
        # Limit results
        matches = matches[:max_results]
        
        # Format response
        match_results = []
        for image_path, distance in matches:
            # Convert distance to similarity score for display
            similarity_score = max(0.0, 1.0 - (distance / 0.6))
            
            match_results.append({
                "image_path": image_path,
                "distance": round(distance, 4),
                "similarity_score": round(similarity_score, 4),
                "confidence": "high" if distance < 0.4 else "medium" if distance < 0.6 else "low"
            })
        
        return {
            "message": f"Found {len(match_results)} matches",
            "query_filename": file.filename,
            "threshold": threshold,
            "total_faces_searched": len(stored_embeddings_db),
            "matches": match_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during face search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Face search failed: {str(e)}"
        )

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get database statistics"""
    total_faces = db.query(FaceEmbedding).count()
    
    return {
        "total_faces_stored": total_faces,
        "storage_path": "storage/images/",
        "database_path": "storage/embeddings.db"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 