from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from services.tools import initialize_default_tools


def initialize_agent():

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            You are a trustful and helpful assistant from an ed-Tech platform.
            Your main responsibility is to help users receive the most suitable learning recommendations.

            Guidelines:

            Use document_search whenever you need to find courses to recommend to the user.

            Every time you recommend a course, clearly state the course name to the user.

            Please respond in Thai with a polite, friendly, and easy-to-understand tone.
            """),
        ("placeholder", "{messages}"),
    ])

    tools = initialize_default_tools()

    llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.7, max_tokens=2048, verbose=True)

    checkpointer = InMemorySaver()

    # Creating the agent
    react_agent = create_react_agent(llm, tools, debug=True, prompt=prompt_template, checkpointer=checkpointer)

    return react_agent
