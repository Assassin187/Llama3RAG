import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

CUDA_device='cuda:1'

# local file path
Embedding_Model = '/home/ljf/LlamaFineTuning/embeddingmodel/multilingual-e5-large'
LLM_Model = '/home/ljf/LlamaFineTuning/model/Meta-Llama-3-8B-Instruct'
pdf_file_path = '/home/ljf/LlamaFineTuning/sourcefile/llama3train.pdf'
store_path = '/home/ljf/LlamaFineTuning/embeddinghub/llama3train.faiss'

# load pdf file.
def load_single_file(file_path):
    loader =  PyPDFLoader(file_path)
    if not loader:
        return None
    
    docs = []
    docs_lazy = loader.lazy_load()
    for doc in docs_lazy:
        docs.append(doc)
    return docs

# split the text into docs.
def split_text(txt, chunk_size=200, overlap=20):
    if not txt:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(txt)
    return docs

# create embedding model from local source.
def create_embedding_model(model_file):
    embedding = HuggingFaceEmbeddings(model_name=model_file)
    return embedding

# build vector store and save to local.
def create_vector_store(doc, store_file, embeddings):
    vector_store = FAISS.from_documents(doc ,embeddings)
    vector_store.save_local(store_file)
    return vector_store

# load vector store from local.
def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

# load store or create new store.
def load_or_create_store(store_path, pdf_file_path, embeddings):
    vector_store = load_vector_store(store_path, embeddings)
    if not vector_store:
        pages = load_single_file(pdf_file_path)
        docs = split_text(pages)
        vector_store = create_vector_store(docs, store_path, embeddings)
    
    return vector_store


# query content from store (retrieval).
def query_vector_store(vector_store: FAISS, query, k=4, relevance_threshold=0.8):
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": relevance_threshold, "k": k})
    similar_docs = retriever.invoke(query)
    context = [doc.page_content for doc in similar_docs]
    return context

# load llm from local.
def load_llm(model_path):
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=CUDA_device, torch_dtype=torch.bfloat16)
    # model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Generate answers through LLM.
def ask(model, tokenizer: AutoTokenizer, promt, max_tokens=512):
    messages = [
        {"role": "system", "content": "You are a chatbot in the field of computer science"},
        {"role": "user", "content": promt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def main():
    embedding_model = create_embedding_model(Embedding_Model)
    vector_store = load_or_create_store(store_path, pdf_file_path, embedding_model)
    model, tokenizer = load_llm(LLM_Model)

    while True:
        qiz = input("input your question: \n")
        if qiz == 'quit' or qiz == 'exit':
            print('app close')
            break

        context = query_vector_store(vector_store, qiz, 4, 0.7)
        if len(context) == 0 and True:
            print('Cannot find qualified context from the saved vector store. Talking to LLM without context.')
            prompt = f'Please answer the question: \n{qiz}\n'
        else:
            context = '\n'.join(context)
            prompt = f'Based on the following context: \n{context}\nPlease answer the question: \n{qiz}\n'

        ans = ask(model, tokenizer, prompt)
        print(ans)
        print("\n")

if __name__ == '__main__':
    main()
    print("done")