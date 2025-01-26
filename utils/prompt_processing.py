import re

def preprocess_query(query: str) -> str:
    query = re.sub(r'[!?@#$%^&*()_+=\[\]{};:"\\|,.<>/?`~\u200B\u200C\u200D\uFEFF]', ' ', query)

    query = re.sub(r'\b\d+[$.,]\b|\b[$.,]\d+\b|\b\d+\.\d+\b', ' ', query)
    
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query