# langchain_intro.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the Model 
# Using ChatOllama for better chat-template support (special tokens like <|eot_id|>)
llm = ChatOllama(
    model="fraud-reasoner",  # fine-tuned Llama-3 model name, registered in Ollama
    temperature=0,
)

# Define our Template
# LangChain abstracts away the Llama-3 specific XML-like tags.
prompt = ChatPromptTemplate.from_template("Question: {input}\nAnswer: Let's think step by step.")

# The Chain (The "Secret Sauce")
# LCEL uses the pipe (|) to send data: Prompt -> Model -> String Parser
chain = prompt | llm | StrOutputParser()

# Execute
# Now 'response' is a clean string, not a complex object
response = chain.invoke({"input": "What is the fraud risk of a transaction from an unknown IP at 3 AM?"})

print("=== final clean resonse ===")
print(response)
