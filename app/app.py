import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
import sys
from app.agents.agent import initialize_agent

def main():
    st.title("🤖 SariSariNews: News from Your Neighborhood")
    agent = initialize_agent()
    user_input = st.text_input("Ask SariSariNews", placeholder="What's the latest news?")

    if st.button("Submit") and user_input:
        with st.spinner("Thinking..."):
            try:
                response = agent.chat(user_input)
                st.write(f"**Response:** {response.response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", "--server.fileWatcherType", "none", sys.argv[0]]
        sys.exit(stcli.main())