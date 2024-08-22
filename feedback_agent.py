from langchain_openai import OpenAI
from langchain.chains import SimpleChain
from langchain.prompts import PromptTemplate

def provide_feedback(user_responses, api_key):
    llm = OpenAI(api_key=api_key, model="gpt-4")
    prompt_template = PromptTemplate(
        input_variables=["responses"],
        template="Provide detailed feedback for the following interview responses:\n\n{responses}"
    )
    chain = SimpleChain(llm=llm, prompt=prompt_template)

    responses_str = "\n".join([f"Q: {resp['question']}\nA: {resp['response']}" for resp in user_responses])
    feedback = chain.run({"responses": responses_str})