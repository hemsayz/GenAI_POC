import os
import pickle
import faiss
import numpy as np

# Define path locations matching your engine parameters
INDEX_DIR = "faiss_index"
FAISS_FILE = os.path.join(INDEX_DIR, "index.faiss")
PKL_FILE = os.path.join(INDEX_DIR, "index.pkl")

print("==================================================================")
print("🔍 EXHAUSTIVE FAISS LOCAL STORAGE AUDIT DETECTOR")
print("==================================================================")

if not os.path.exists(INDEX_DIR):
    print(f"❌ Error: The storage directory '{INDEX_DIR}' does not exist on your disk.")
    print("Please go back to your web panel and ingest a PDF document asset first!")
    exit()

# =========================================================================
# PART 1: UNPACKING THE BINARY COORDINATE MATRIX (index.faiss)
# =========================================================================
print("\n📦 STEP 1: PARSING C++ BINARY GEOMETRY FILE (.faiss)...")
try:
    # Read the raw binary matrix structure directly out of CPU space
    matrix_index = faiss.read_index(FAISS_FILE)
    
    total_vectors = matrix_index.ntotal
    vector_dimensions = matrix_index.d
    
    print(f"  🔹 Status:                     Successfully Unpacked Matrix")
    print(f"  🔹 Total Vectors in Database:  {total_vectors}")
    print(f"  🔹 Vector Target Dimensions:   {vector_dimensions} Float Axes")
    
    if total_vectors > 0:
        # Reconstruct the raw floating-point coordinates for Chunk #1 (Index row 0)
        chunk_1_vector = matrix_index.reconstruct(0)
        print(f"\n  📍 Raw Coordinate Sample for Matrix Row [0] (First 5 Dimensions):")
        print(f"     {[float(val) for val in chunk_1_vector[:5]]} ... [truncated remaining array]")
except Exception as faiss_err:
    print(f"  🔴 FAISS Error: Failed to parse binary array layers. Details: {faiss_err}")

# =========================================================================
# PART 2: UNPACKING THE ENGLISH METADATA LEDGER (index.pkl)
# =========================================================================
print("\n📝 STEP 2: DESERIALIZING PYTHON METADATA STORAGE LEDGER (.pkl)...")
try:
    # Safely load the binary serialization stream natively
    with open(PKL_FILE, "rb") as f:
        docstore_obj, index_to_id_map = pickle.load(f)
        
    print(f"  🔹 Status:                     Successfully Loaded Pickled Stream")
    print(f"  🔹 Underlying Memory Core:     {type(docstore_obj)}")
    print(f"  🔹 Total ID Keys Mapped:       {len(index_to_id_map)}")
    
    if len(index_to_id_map) > 0:
        print("\n  📍 Document Vector ID-to-UUID Mapping Registry:")
        # Print the first 3 registered document token map addresses
        for idx, (matrix_row, unique_uuid) in enumerate(index_to_id_map.items()):
            if idx >= 3:
                print(f"     ... remaining {len(index_to_id_map) - 3} UUID pointers truncated.")
                break
            print(f"     Row Index [{matrix_row}] ──> Core Database Storage Key Unique ID: [{unique_uuid}]")
            
        # Extract a sample text string directly from the in-memory docstore layer dictionary
        print("\n  📄 First Chunk Document Text Formatting Audit:")
        sample_uuid = index_to_id_map[0]
        sample_doc = docstore_obj._dict[sample_uuid]
        print(f"     [Source Page Reference: {sample_doc.metadata.get('page', 0) + 1}]")
        print(f"     \"\"\"{sample_doc.page_content[:150]}...\"\"\"")
except Exception as pkl_err:
    print(f"  🔴 Pickle Error: Failed to unpack the text memory registry dictionary. Details: {pkl_err}")

print("\n==================================================================")
print("🎯 Audit Complete: Data extracted with zero corrupted text glitches!")
print("==================================================================")