from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')


prompt = PromptTemplate(
    template = 'Answer the folowing Question \n {question} from the following {text}',
    input_variables={'question' , 'text'}
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdvq4hn-a/p/itm4b244ba3a72ce?pid=COMHZQX4X7QDXECT&lid=LSTCOMHZQX4X7QDXECT78YQDA&marketplace=FLIPKART&q=apple+m5+laptop&store=6bo%2Fb5g&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_8_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_8_na_na_na&fm=organic&iid=797828b7-00d4-430c-9993-fd0a7ffdf06d.COMHZQX4X7QDXECT.SEARCH&ppt=None&ppn=None&ssid=x18pbs4we80000001787659809397&qH=1571987bb0056ad6&ov_redirect=true'
loader = WebBaseLoader(url)



chain  = prompt | model | parser

docs = loader.load()

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))
