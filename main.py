__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import os
import warnings
from datetime import datetime

import streamlit as st
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
import logging

logging.getLogger('streamlit').setLevel(logging.ERROR)
load_dotenv()
os.getenv("OPENAI_API_KEY")
from services import StreamlitCallbackHandler, initialize_agent


def write_logfile(log_dict, path=r"C:\Users\User\Desktop\conicle-agent\services\logs\agent_workflow.log"):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=2)
        f.write("\n\n")  # Separate entries with blank line

def log_agent_response(response, llm_model="gpt-4", agent_type="zero-shot-react-description"):
    """
    Extracts and formats a log from the agent response using LangGraph output.
    
    Parameters:
        response (dict): Agent output, such as from LangGraph.
        llm_model (str): The LLM model used.
        agent_type (str): The agent strategy/type used.
    
    Returns:
        dict: Formatted log object for storing or inspection.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    # Initialize the steps_log list to store each step's log
    steps_log = []
    final_output = ""
    tools_used = set()

    # Iterate over the 'messages' to extract necessary data
    for i, message in enumerate(response['messages']):
        if isinstance(message, AIMessage):
            # Collect content and tool calls from AIMessage
            final_output = message.content  # Last AI message is the final output
            for tool_call in message.tool_calls:
                tools_used.add(tool_call["name"])  # Track tools used
            
            # Add relevant step information
            steps_log.append({
                "step_number": i + 1,
                "thought": message.content, 
                "tool_calls": message.tool_calls
            })
        
        elif isinstance(message, ToolMessage):
            # Track tool message content and tool call ID
            steps_log.append({
                "step_number": i + 1,
                "tool_name": message.name,
                "tool_output": message.content
            })

    # Construct the agent log based on the provided structure
    agent_log = {
        "timestamp": datetime.now().isoformat(),
        "input": response['messages'][0].content,  # First message is the input
        "final_output": final_output,
        "intermediate_steps": steps_log,
        "llm_model": llm_model,
        "tools_used": list(tools_used),
        "agent_type": agent_type,
        "status": "success"
    }

    return agent_log

def chat_completion(prompt, callback_handler, max_retries=3):
    attempts = 0
    while attempts < max_retries:
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "human", "content": prompt}]},
            # {"chat_history": []},  # optionally add chat_history here
            {"callbacks": [callback_handler], "configurable": {"thread_id": "thread-1"}},
        )

        if response['messages'][-1].content != "Oops! Something went wrong. Please give it another try!":
            return response
        attempts += 1
    return "Failed after several attempts. Please try again later."

# Initialize Streamlit app
st.title("Chat with Data")

# Initial assistant message
initial_message = """
สวัสดีครับ/ค่ะ! ยินดีต้อนรับสู่ผู้ช่วยอัจฉริยะสำหรับการวิเคราะห์ Social Listening และการวิเคราะห์ข้อมูลธุรกิจ

ที่นี่ คุณสามารถค้นหาข้อมูลเชิงลึกและแนวโน้มต่าง ๆ ที่เกี่ยวกับ:
- **แบรนด์** (เช่น Samsung, BYD) 
- **วันและเวลา** (เจาะจงวันที่ เช่น 2024-07-04 หรือช่วงเวลา)
- **หมวดหมู่** (เช่น การตลาด, รถยนต์)
- **แพลตฟอร์ม** (เช่น Twitter, Instagram)
- **เอกสารเชิงลึกด้านธุรกิจ** ซึ่งรวมถึงรายงานการวิเคราะห์กลยุทธ์ การประเมินผลการดำเนินงานรายเดือน และคู่มือสำหรับอุตสาหกรรมต่าง ๆ

**ตัวอย่างคำถามที่คุณอาจสนใจ:**
- "ข้อมูลการมีส่วนร่วมของแบรนด์ Samsung ในวันที่ 2024-07-04 เป็นอย่างไร?"
- "ในเดือนกรกฎาคม มีการพูดถึงพรรคเพื่อไทยมากแค่ไหน?"
- "มีความคิดเห็นอะไรบ้างเกี่ยวกับรถยนต์ BYD บน Instagram?"
- "มีข้อมูลสำคัญอะไรบ้างจากเอกสารแนวทางการเติบโตในธุรกิจสุขภาพ?"
- "สรุปกลยุทธ์และผลการดำเนินงานของ Samsung ไตรมาส 3 คืออะไร?"

คุณสามารถพิมพ์คำถามของคุณด้านล่างนี้ และฉันจะช่วยค้นหาข้อมูลที่ตรงกับความต้องการของคุณ! 
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "default_thread"

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input and chat handling
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback_handler = StreamlitCallbackHandler(st.empty())
        with st.spinner("กำลังประมวลผล..."):

            response = chat_completion(prompt, callback_handler)

        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response['messages'][-1].content}])
        
        # For debug and log agent step thinking.
        agent_log = log_agent_response(response)
        write_logfile(agent_log)
        
else:
    with st.chat_message("assistant"):
        st.markdown(initial_message)
