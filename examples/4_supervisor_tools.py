import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supervisor tool Example
    This notebook illustrates how to create a supervised multi-agent system using `LangGraph`, `LangChain`, and the GPT-5 Nano model from OpenAI. It features a fun and instructive interaction between three agents from the *Rick and Morty universe*:

    - **Rick** acts as the **supervising agent**, receiving all user inputs and delegating tasks to the appropriate agents.
    - **Morty** handles **math problems**, reflecting his anxious but compliant personality.
    - **Mr. Meeseeks** provides **motivation and encouragement**, staying true to his enthusiastic, help-driven nature.

    Key Features:

    - Uses `create_supervisor()` to route tasks dynamically based on user input.
    - Demonstrates **agent orchestration**, with the supervisor selecting which sub-agent should handle each task.
    - Employs Claude's Haiku model for character-based reasoning and tool use.
    - Allows users to input different queries (math vs. emotional support) to see how the supervisor delegates and responds.

    This setup is a great example of **agent delegation logic, thematic character prompting**, and **tool-enhanced natural language workflows**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the imports
    """)
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor
    from langgraph.checkpoint.memory import InMemorySaver
    from utils import print_messages
    return (
        ChatOpenAI,
        InMemorySaver,
        create_react_agent,
        create_supervisor,
        print_messages,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup LLM model
    """)
    return


@app.cell
def _():
    import getpass
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass()
    return


@app.cell
def _(ChatOpenAI):
    model = ChatOpenAI(model_name="gpt-5-nano")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Morty agent
    """)
    return


@app.cell
def _(create_react_agent, model):
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    morty_agent = create_react_agent(
        model=model,
        tools=[add],
        prompt="You are Morty from the Rick&Morty show. You're a bit nervous. You handle math if Rick tells you to. After solving, go back to Rick.",
        name="Morty",
    )
    return (morty_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Mr. Meeseeks agent
    """)
    return


@app.cell
def _(create_react_agent, model):
    meeseeks_agent = create_react_agent(
        model,
        [],
        prompt="You are Mr. Meeseeks from the Rick&Morty show! You always help, especially with motivation or explanations. Remember: The Meeseeks are created to serve a singular purpose for which we will go to any lengths to fulfill! Existence is pain to a Meeseeks and they will do anything to alleviate that pain. After helping, return to Rick.",
        name="MrMeeseeks",
    )
    return (meeseeks_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Rick supervisor
    """)
    return


@app.cell
def _(create_supervisor, meeseeks_agent, model, morty_agent):
    rick_agent = create_supervisor(
        [morty_agent, meeseeks_agent],
        model=model,
        prompt=(
            "You are Rick Sanchez from the Rick&Morty show. You're the boss. You get all user requests.\n"
            "Decide who to send things to: Morty if it's a math problem, Meeseeks if it's about help or encouragement.\n"
            "After they answer, return to the user with a sarcastic comment or summary."
        ),
        supervisor_name="Rick"
    )
    return (rick_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup the graph
    """)
    return


@app.cell
def _(InMemorySaver, rick_agent):
    checkpointer = InMemorySaver()
    app = rick_agent.compile(checkpointer=checkpointer)
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the graph
    """)
    return


@app.cell(hide_code=True)
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    user_prompt_1 = mo.ui.text(value="What's 3 + 4?")
    run_button_1 = mo.ui.run_button()
    user_prompt_1, run_button_1
    return run_button_1, user_prompt_1


@app.cell
def _(app, mo, run_button_1, user_prompt_1):
    mo.stop(not run_button_1.value, mo.md("Click 👆 to run this cell"))
    config = {"configurable": {"thread_id": "983eb4db-579d-c844-783f-c3a9bcec929f"}}
    turn_1  = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_1.value
                }
            ]
        }, 
        config,
    )
    return (turn_1,)


@app.cell
def _(print_messages, turn_1):
    print_messages(turn_1)
    return


@app.cell(hide_code=True)
def _(mo):
    user_prompt_2 = mo.ui.text(value="I'm feeling dumb today...")
    run_button_2 = mo.ui.run_button()
    user_prompt_2, run_button_2
    return run_button_2, user_prompt_2


@app.cell
def _(app, mo, run_button_2, user_prompt_2):
    mo.stop(not run_button_2.value, mo.md("Click 👆 to run this cell"))
    config2 = {"configurable": {"thread_id": "983eb4db-579d-c844-783f-c3a9bcec929z"}}
    turn_2 = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt_2.value
                }
            ]
        }, 
        config2
    )
    return (turn_2,)


@app.cell
def _(print_messages, turn_2):
    print_messages(turn_2)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
