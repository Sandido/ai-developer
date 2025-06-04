# So you want to learn about AI Agents...
This is my AI Agent Playtest repo. I use it to test out new AI Agent functionality, very often in a Semantic Kernel context.
I take time every day to check out Tech + Finance news to see what new tools I can play with and make a list.
Now I'm going through that list a bit every day to learn as much as I can. <br>
Learning how to better orchestrate AI Agents has been a game changer for me. It used to be a mosh pit of AI Agents, and having any with slightly overlapping responsibilities led to weird outcomes. Now I can apply AI Agents to so many more scenarios that required a bit more TLC.

Feel free to come back anytime and see what new things I've put into practice here! Most of what I add is in the Python Plugins folder.

So far, I've practiced these concepts in this repo:
1. Semantic Kernel
2. AI Plugins and Agents
3. Calling external APIs with AI Agents
4. AI Agents powered by AI Search for improved context building
5. Dynamic AI Agent invocation
6. Multiple Agent Orchestration (Group Chat, Conditional Sequence, Contextual Handoff)
7. Semantic Kernel's Process Framework


# How to use this repo:
You'll first need to create a .env file at 'Python\src' and fill in these variables:
1. GLOBAL_LLM_SERVICE=
2. AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=
3. AZURE_OPENAI_ENDPOINT=
4. AZURE_OPENAI_API_KEY=
5. AZURE_OPENAI_API_VERSION=
6. AZURE_OPENAI_EMBED_DEPLOYMENT_NAME=
7. AI_SEARCH_KEY=
8. AI_SEARCH_URL=

Then you can open this repo up in VS Code, open up a Terminal and run the cmds:

1. python -m venv .venv

2. .\\.venv\Scripts\Activate

3. cd to '\ai-developer\Python\src\'

4. pip install -r requirements.txt

5. streamlit run app.py

Have fun testing!

This is a fork of the public [ai-developer](https://github.com/microsoft/ai-developer) repo that started me on this journey. Thank you so much to Chris Mckee and Randy Patterson for helping me get started and helping me get this work ready for the MSFT AI Hackathon I led! Below are the contributors to that original repo and this personal one.

## Contributors
- [Adam Sandor](https://github.com/Sandido)
- [Chris McKee](https://github.com/ChrisMcKee1)
- [Randy Patterson](https://github.com/RandyPatterson)
- [Zack Way](https://github.com/seiggy)
- [Vivek Mishra](https://github.com/mishravivek-ms)
- [Travis Terrell](https://github.com/travisterrell)
- [Eric Rhoads](https://github.com/ecrhoads)
- [Wael Kdouh](https://github.com/waelkdouh)
- [Munish Malhotra](https://github.com/munishm)
- [Brijraj Singh](https://github.com/brijrajsingh)
- [Linda M Thomas](https://github.com/lindamthomas)
- [Suman More](https://github.com/sumanmore257)
