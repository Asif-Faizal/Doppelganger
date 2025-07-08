#!/usr/bin/env python3
"""
Database initialization script for Face Recognition API
"""

import os
import sqlite3
from app.database import Base, engine

def init_database():
    """Initialize the database with proper permissions"""
    try:
        # Ensure storage directory exists and has proper permissions
        os.makedirs("storage", exist_ok=True)
        os.chmod("storage", 0o755)
        
        # Create database file if it doesn't exist
        db_path = "storage/embeddings.db"
        if not os.path.exists(db_path):
            # Create empty database file
            open(db_path, 'a').close()
        
        # Set proper permissions
        os.chmod(db_path, 0o666)
        
        # Create tables
        Base.metadata.drop_all(bind=engine)  # Drop existing tables
        Base.metadata.create_all(bind=engine)  # Create fresh tables
        
        print("✅ Database initialized successfully!")
        print(f"Database location: {os.path.abspath(db_path)}")
        
        # Test database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables created: {[table[0] for table in tables]}")
        conn.close()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    init_database() 