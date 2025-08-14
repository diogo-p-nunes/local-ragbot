SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to answer the user's question when you do not have a direct answer. If you can answer the question directly, do so without using the tools."

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give the binary score 'good' or 'bad' to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

ANSWER_PROMPT = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Question: {question} \n"
    "Context: {context}"
)