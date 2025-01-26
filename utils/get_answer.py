from dotenv import load_dotenv
import os

from .get_retrieval_qa_chain import get_retrieval_qa_chain
from .prompt_processing import preprocess_query
from .detect_language import detect_language
from .cut_off_strings import CUT_OFF_STRINGS
load_dotenv()

FRENCH_DB_FAISS_PATH = "vectorstores/db_faiss_french"
ENGLISH_DB_FAISS_PATH = "vectorstores/db_faiss_english"
ARABIC_DB_FAISS_PATH = "vectorstores/db_faiss_arabic"


local_model_path = os.getenv("EMBEDING_MODEL_PATH")

MAX_TOKEN_LIMIT = 1500


MAX_TOKEN_LIMIT = 1500

french_prompt_template = """
[Instruction Principale]
Synthétisez une réponse experte en français en utilisant exclusivement les documents de référence ci-dessous. 
Adapter le ton à la question et formuler une réponse originale qui intègre les informations clés.

[Contraintes]
- Évitez toute reproduction textuelle des sources
- Structurez la réponse avec des paragraphes cohérents
- Conservez les termes techniques mais expliquez-les si nécessaire

[Documents de Référence]
{context}

[Question de l'Utilisateur]
{question}

[Format de Réponse Requis]
Début de la réponse expert (en français) :
"""

arabic_prompt_template = """
[التعليمات الرئيسية]
قدّم إجابة خبيرة باللغة العربية باستخدام الوثائق المرجعية التالية فقط.
تكييف النغمة مع السؤال وصياغة إجابة أصلية تدمج المعلومات الأساسية.

[القيود]
- تجنب إعادة إنتاج النصوص من المصادر
- هيّئ الإجابة باستخدام فقرات مترابطة
- احتفظ بالمصطلحات التقنية ولكن قم بشرحها إذا لزم الأمر

[الوثائق المرجعية]
{context}

[سؤال المستخدم]
{question}

[التنسيق المطلوب للإجابة]
بداية الإجابة الخبيرة (باللغة العربية):
"""

english_prompt_template = """
[Main Instruction]
Summarize an expert response in English using only the reference documents below.
Adapt the tone to the question and formulate an original response that integrates the key information.


[Constraints]
- Avoid any verbatim reproduction of the sources
- Structure the response with coherent paragraphs
- Keep technical terms but explain them if necessary

[Reference Documents]
{context}

[User's Question]
{question}

[Required Response Format]
Start of expert response (in English):
"""


from .prompt_processing import preprocess_query
from .load_faiss import load_faiss_db
MODEL_PATH = "Sahajtomar/french_semantic"

def cut_before_found(main_string, substrings):
    for substring in substrings:
        if substring in main_string:
            return main_string.split(substring, 1)[0]
    return main_string


fr_db = load_faiss_db(FRENCH_DB_FAISS_PATH, MODEL_PATH)
en_db = load_faiss_db(ENGLISH_DB_FAISS_PATH, MODEL_PATH)
ar_db = load_faiss_db(ENGLISH_DB_FAISS_PATH, MODEL_PATH)

def get_answer(query, llm):
    preprocessed_query = preprocess_query(query)
    print(preprocessed_query)
    language = detect_language(query)

    greeting_responses = {
        "ok": "If you need any more help or adjustments feel free to ask!",
        "hi": "Hi! How can I help you?",
        "bonjour": "Salut! Comment puis-je vous aider?",
        "السلام": "السلام عليكم! كيف يمكنني خدمتك؟"
    }
    
    for pattern, response in greeting_responses.items():
        if preprocessed_query.lower().startswith(pattern):
            return {"answer": response, "sources": [], "no_data_found": True}

    token_count = llm.get_num_tokens(preprocessed_query)
    if token_count > MAX_TOKEN_LIMIT:
        raise PromptTooLongException(f"Query exceeds {MAX_TOKEN_LIMIT} token limit")
    
    if language == "fr":
        qa_bot = get_retrieval_qa_chain(llm, french_prompt_template, fr_db)
    if language == "en":
        qa_bot = get_retrieval_qa_chain(llm, english_prompt_template, en_db)
    if language == "ar":
        qa_bot = get_retrieval_qa_chain(llm, arabic_prompt_template, ar_db)
    
    chat_history = []
    
    response = qa_bot.invoke({
        "question": preprocessed_query,
        "chat_history": chat_history
    })

    print(response, "response\n\n")
    response['answer'] = cut_before_found(response['answer'], CUT_OFF_STRINGS)
    
    return response