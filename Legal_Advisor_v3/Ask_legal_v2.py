from langchain_google_firestore import FirestoreVectorStore
import vertexai
from vertexai.language_models import ChatModel
from langchain_google_vertexai import VertexAIEmbeddings
from vertexai.generative_models import GenerativeModel, GenerationConfig


def query_llm(prompt, project_id, model_name, embed_model_name, collection, location, max_output_tokens, temperature, top_p, context):
    vertexai.init(project=project_id, location=location)
    chat_model = ChatModel.from_pretrained(model_name)
    embedding = VertexAIEmbeddings(
        model_name=embed_model_name,
        project=project_id,
    )
    vector_store = FirestoreVectorStore(collection=collection, embedding_service=embedding)
    
    parameters = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    chat = chat_model.start_chat(
        context=context
    )

    matches = vector_store.similarity_search(prompt, 5)
    #vector_store.max_marginal_relevance_search(prompt, 5)

    documents = [doc.page_content for doc in matches]

    extended_prompt = f"""
{context}

Normatividad vigente en Colombia:
{' '.join(documents)}

Pregunta:
{prompt}
"""

    response = chat.send_message(extended_prompt, **parameters)
    return response.text


def query_llm_extended(prompt, project_id, model_name, embed_model_name, collection, location, max_output_tokens, temperature, top_p, context):
    vertexai.init(project=project_id, location=location)
    chat_model = ChatModel.from_pretrained(model_name)
    embedding = VertexAIEmbeddings(
        model_name=embed_model_name,
        project=project_id,
    )
    vector_store = FirestoreVectorStore(collection=collection, embedding_service=embedding)
    
    parameters = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    chat = chat_model.start_chat(
        context=context
    )

    matches = vector_store.similarity_search(prompt, 5)
    #vector_store.max_marginal_relevance_search(prompt, 5)

    documents = [doc.page_content for doc in matches]

    extended_prompt = f"""
{context}

Normatividad vigente en Colombia:
{' '.join(documents)}

Pregunta:
{prompt}
"""

    response = chat.send_message(extended_prompt, **parameters)
    output = f"""Respuesta: {response.text}

Extended Prompt:
{extended_prompt}"""
    return output


def init_gemini(prompt, project_id, model_name, location, max_output_tokens, temperature, top_p, system_instruction):
    '''This function set system instructions and send an initialization prompt with user data'''
    vertexai.init(project=project_id, location=location)
    generation_config = GenerationConfig(max_output_tokens=max_output_tokens, temperature=temperature, top_p=top_p)    
    chat_model = GenerativeModel(model_name, generation_config=generation_config, system_instruction=system_instruction)
    chat = chat_model.start_chat() ## Here we can include chat history
    response = chat.send_message(prompt)
    output = response.text
    return chat, output


def query_gemini(chat, prompt, embed_model_name, project_id, collection):
    '''This function send user prompt including vector search retrieval'''
    embedding = VertexAIEmbeddings(model_name=embed_model_name, project=project_id)
    vector_store = FirestoreVectorStore(collection=collection, embedding_service=embedding)

    matches = vector_store.similarity_search(prompt, 5)
    #vector_store.max_marginal_relevance_search(prompt, 5)

    documents = [doc.page_content for doc in matches]

    extended_prompt = f"""

Internal Documents:
{' '.join(documents)}

Question:
{prompt}
"""

    response = chat.send_message(extended_prompt)
    output = response.text
    return chat, output, extended_prompt