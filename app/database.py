import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Create storage directory if it doesn't exist
os.makedirs("storage", exist_ok=True)

# Ensure storage directory has proper permissions
os.chmod("storage", 0o755)

# Database setup
DATABASE_URL = "sqlite:///storage/embeddings.db"
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for debugging SQL queries
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, index=True)
    embedding = Column(Text)  # JSON string of the embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def set_embedding(self, embedding_vector):
        """Convert numpy array to JSON string for storage"""
        self.embedding = json.dumps(embedding_vector.tolist())
    
    def get_embedding(self):
        """Convert JSON string back to numpy array"""
        import numpy as np
        return np.array(json.loads(self.embedding))

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 