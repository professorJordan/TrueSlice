
import numpy as np
import faiss
import ollama
import configparser
from sentence_transformers import SentenceTransformer


############################################################################################################################################

#pull in the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

#Path to answer file in txt format. We need it for data retrievel
LongAnswerFile = config["QueryIndex"]["LongAnswerFile"]

#path to FAISS Vector DB Index. You probably called it NewIndex in the creation script
IndexFile = config["QueryIndex"]["IndexFile"]

#path to Questions file in txt format. We need it to query the RAG
QuestionFile = config["QueryIndex"]["QuestionFile"]

#path to where you want to store the short answers to the questions.
OutputFile = config["QueryIndex"]["ResponseFile"]

#select the model you want to use
Model = config["QueryIndex"]["Model"]

#Retrieve the instruction to be passed to the model
SystemInstruction = config["QueryIndex"]["SystemInstruction"]

############################################################################################################################################

# Load the FAISS index, the original answer file, and the file containing the questions

index = faiss.read_index(IndexFile)
#index.nprobe = 20  # Search 20 of the 300 clusters for strong recall

# The original text of the answers
with open(LongAnswerFile, "r", encoding="utf-8") as file:
    documents = [line.strip() for line in file if line.strip()]

# Open the question file and load the questions
with open(QuestionFile, "r", encoding="utf-8") as file:
    Questions = [line.strip() for line in file if line.strip()]

############################################################################################################################################

# for each question in the question file, do RAG and then write the answer to the OutputFile

#Encode and normalize the user's question
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
for user_question in Questions:
    print("\n###################################################################")    
    print("RAG Question:")
    print(user_question + "\n")

    #Encode and normalize the user's question
    #embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    query_vector = np.array(embed_model.encode([user_question])).astype('float32')
    faiss.normalize_L2(query_vector)

    # 3. Retrieve top k=3 matching document IDs
    k = 5
    scores, indices = index.search(query_vector, k)

    ############################################################################################################################################

    # Augment: Map IDs back to text and build the context block

    retrieved_chunks = []
    for rank, idx in enumerate(indices[0]):
        if idx != -1:  # -1 indicates no match found
            retrieved_chunks.append(documents[idx])
            print(f"[Match {rank+1} | Score: {scores[0][rank]:.4f}]: {documents[idx]}")

    context = "\n- ".join(retrieved_chunks)


    prompt = f"""Choices:
    - {context}
    Query: {user_question}"""

    ############################################################################################################################################

    # Generate: Send the augmented prompt to local Mistral via Ollama

    print("\nGenerating answer with model...\n")
    response = ollama.chat(
        model=Model,
        messages=[{"role": "system", "content": SystemInstruction},{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 15}
    )
    print("RAG Answer:")
    print(response["message"]["content"])

    ############################################################################################################################################
    
    # Write answer to output file
    clean_answer = " ".join(response["message"]["content"].splitlines()).strip()
    with open(OutputFile, "a", encoding="utf-8") as out_file:
        out_file.write(clean_answer + "\n")





