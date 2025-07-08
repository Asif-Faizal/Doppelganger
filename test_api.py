#!/usr/bin/env python3
"""
Simple test script for Face Recognition API

This script demonstrates how to use the API endpoints.
Make sure the API server is running before executing this script.
"""

import requests
import json
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running successfully!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_upload_images(image_paths):
    """Test uploading images to extract embeddings"""
    print(f"\n📤 Testing upload with {len(image_paths)} images...")
    
    try:
        files = []
        for path in image_paths:
            if not Path(path).exists():
                print(f"❌ Image file not found: {path}")
                return False
            files.append(('files', open(path, 'rb')))
        
        response = requests.post(f"{API_BASE_URL}/upload-images", files=files)
        
        # Close file handles
        for _, file_handle in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"Processed: {result['message']}")
            for item in result['results']:
                if item['status'] == 'success':
                    print(f"   - {item['filename']}: ✅ (ID: {item['embedding_id']})")
                else:
                    print(f"   - {item['filename']}: ❌ {item['message']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_search_face(image_path, threshold=0.6):
    """Test face search functionality"""
    print(f"\n🔍 Testing face search with distance threshold {threshold}...")
    
    try:
        if not Path(image_path).exists():
            print(f"❌ Query image not found: {image_path}")
            return False
        
        files = {'file': open(image_path, 'rb')}
        data = {'threshold': threshold, 'max_results': 10}
        
        response = requests.post(f"{API_BASE_URL}/search-face", files=files, data=data)
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Search successful!")
            print(f"Query: {result['query_filename']}")
            print(f"Searched {result['total_faces_searched']} faces in database")
            
            if result['matches']:
                print(f"Found {len(result['matches'])} matches:")
                for match in result['matches']:
                    print(f"   - {match['image_path']}: distance={match['distance']:.4f}, similarity={match['similarity_score']:.4f}, confidence={match['confidence']}")
            else:
                print("No matches found within the distance threshold")
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_stats():
    """Test getting database statistics"""
    print("\n📊 Testing stats endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            result = response.json()
            print("✅ Stats retrieved successfully!")
            print(f"Total faces stored: {result['total_faces_stored']}")
            print(f"Storage path: {result['storage_path']}")
            print(f"Database path: {result['database_path']}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Face Recognition API Test Suite")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Cannot proceed with tests. Please start the API server first:")
        print("   python run.py")
        sys.exit(1)
    
    # Test 2: Stats (before uploading)
    test_stats()
    
    # For demo purposes, we'll show how to test with images
    print("\n📝 To test with actual images:")
    print("1. Place some face images in the current directory")
    print("2. Uncomment and modify the test calls below")
    print("3. Run this script again")
    
    # Example test calls (uncomment and modify paths as needed):
    # test_upload_images(["face1.jpg", "face2.jpg"])
    # test_search_face("query_face.jpg", threshold=0.6)
    # test_stats()  # Check stats after upload
    
    print("\n✅ Basic API connectivity test completed!")
    print("\n🔗 Visit http://localhost:8000/docs for interactive API documentation")

if __name__ == "__main__":
    main() 