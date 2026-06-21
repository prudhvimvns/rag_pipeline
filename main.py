from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()
core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"Core version: {core_version}")
print(f"LangGraph version: {lg_version}")

def main():
    print("Hello from rag-project!")

    #Test anthropic
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    response = llm.invoke("Say 'setup complete' in one word")
    print(f"Anthropic response: {response}")

    #Test google genai
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    # response = llm.invoke("Say 'setup complete' in one word")
    # print(f"Google GenAI response: {response}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
