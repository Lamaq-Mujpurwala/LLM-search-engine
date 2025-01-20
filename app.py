import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper , DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun , DuckDuckGoSearchRun
from langchain.agents import AgentExecutor ,create_openai_tools_agent , initialize_agent , AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\lamaq\OneDrive\Desktop\GENAI\.env")
key = os.getenv("GROQ_API_KEY")


wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2 , doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

ddg_search = DuckDuckGoSearchRun(name="Search")

tools = [wiki , arxiv , ddg_search]


st.title("LLM Powered by serach capabilities!!")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role" : "assistant" , "content" : "Hi How can i Help you today?" },
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if prompt:=st.chat_input(placeholder="What is Meaning of Life?"):
    st.session_state.messages.append({"role" : "user" , "content" : prompt})
    st.chat_message("user").write(prompt)
    
    llm  = ChatGroq(
    api_key=key,
    model="llama-3.3-70b-versatile",
    max_retries=2,
    temperature=1,
    streaming=True
    )
    
    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , handling_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages , callbacks=[st_callback])
        st.session_state.messages.append({"role" : "assistant" , "content" : response})
        st.write(response)
