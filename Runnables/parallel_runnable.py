from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Write Pros of {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write cons of {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

parser = StrOutputParser()


chain = RunnableParallel({
    'pro' : RunnableSequence(prompt1, model, parser),
    'con' : RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({'topic':'AI'})

print(result['pro'])
print(result['con'])
