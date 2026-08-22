from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda, RunnablePassthrough , RunnableBranch
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template= 'Write detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

parser = StrOutputParser()

report_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableParallel({'joke' : prompt2 | model | parser, 'word_count': RunnableLambda(word_count)})),
    (lambda x: len(x.split())<500, RunnableParallel({'report' : RunnablePassthrough(), 'word_count': RunnableLambda(word_count)})),

    RunnablePassthrough()
)

final_chain = report_chain | branch_chain

print(final_chain.invoke({'topic': 'Stye'}))

