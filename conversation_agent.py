from langchain.chains import ConversationChain 
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

def conduct_interview(questions, memory, api_key):
    llm = OpenAI(api_key=api_key, model="gpt-4")
    chain = ConversationChain(llm=llm, memory=memory)
    
    user_responses = []
    for question in questions:
        print(f"Interviewer: {question}")
        response = input("Your Answer: ")
        user_responses.append({"question": question, "response": response})
        chain.run(input=question)
        chain.run(input=response)
    return user_responses
