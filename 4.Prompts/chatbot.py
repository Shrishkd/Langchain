from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')

hist = []

while True:
    user_input = input("You: ")
    hist.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(hist)
    hist.append(result)
    print("AI: ", result.content[0]["text"])

print(hist)