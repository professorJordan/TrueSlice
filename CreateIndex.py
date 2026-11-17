import numpy as np
import configparser
import faiss
from sentence_transformers import SentenceTransformer

#pull in the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

#Provide the path to the text data file you want to ingest into the FAISS Vector Index
LongAnswerFile = config["CreateIndex"]["LongAnswerFile"]

#Define the path where you want to store the FAISS Vector Index
NewIndex = config["CreateIndex"]["NewIndexFile"]
 
# 1. Access Data Set
with open(LongAnswerFile, "r", encoding="utf-8") as file:
    documents = [line.strip() for line in file if line.strip()]

print(f"Loaded {len(documents)} documents.")
 
# 2. Deploy the Large Model (Embedding Generator)
# We use a highly rated, lightweight model. You can swap this with larger models like 'all-mpnet-base-v2'
print("Loading the embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# 3. Generate Vectors
print("Generating embeddings for the dataset...")
database_vectors = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True
)
 
database_vectors = np.array(database_vectors).astype('float32')
faiss.normalize_L2(database_vectors)
dimension = database_vectors.shape[1]  # Dimensionality of the model's output (384 for MiniLM)

# 4. Configure the IVFFlat index
num_docs = len(documents)
if num_docs < 5000:
    index = faiss.IndexFlatIP(dimension)
else:
    nlist = min(300, max(1, num_docs // 39))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(database_vectors)
    
# 5. Add vectors to trained index 
index.add(database_vectors)
print(f"Total vectors indexed: {index.ntotal}")

# 6. Save index to disk
faiss.write_index(index, NewIndex)

