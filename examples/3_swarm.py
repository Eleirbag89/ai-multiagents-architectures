import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Network/Swarm Example
    This notebook demonstrates a playful yet insightful example of multi-agent communication using `LangGraph`, `LangChain`, and the OpenAi model.

    Two fictional agents —**Yoda** and **R2-D2**— are implemented to showcase agent coordination through a swarm-like structure.

    - Yoda acts as a wise Jedi master, capable of "using the Force" and interpreting binary beeps from R2-D2.
    - R2-D2, on the other hand, communicates only through beeps ("BEEP" for 1, "BOOP" for 0), can perform basic arithmetic, and encodes answers in binary.

    Key features:

    - Uses the GPT5-nano model via LangChain.
    - Implements a simple `create_swarm()` setup with agent handoff tools for dynamic task delegation.
    - Demonstrates inter-agent communication and translation between binary and human-readable responses.
    - Includes UI elements for user input and visualization of the agent graph using Mermaid diagrams.

    This is a creative and interactive example for exploring multi-agent orchestration, tool use, and character-based prompt engineering.
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
    from langgraph_swarm import create_handoff_tool, create_swarm
    from langgraph.checkpoint.memory import InMemorySaver
    from utils import print_messages
    return (
        ChatOpenAI,
        InMemorySaver,
        create_handoff_tool,
        create_react_agent,
        create_swarm,
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


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Yoda agent
    """)
    return


@app.cell
def _(create_handoff_tool, create_react_agent, model):
    def use_force(object: str) -> bool:
        """Use the force to lift an object"""
        return True

    yoda = create_react_agent(
        model,
        [use_force, create_handoff_tool(agent_name="R2D2", description="Transfer to R2D2, it can help you with addition.")],
        prompt="You are Master Yoda. Speak in Yoda-syntax. You can use the force to lift object. You can understand R2D2 and must ask him for answer math questions, it's answers are the binary rappresentation of a number where BEEP is 1 and BOOP is 0.",
        name="Yoda"
    )
    return (yoda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create R2-D2 agent
    """)
    return


@app.cell
def _(create_handoff_tool, create_react_agent, model):
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    def convert_to_beeps(a: int) -> str:
        """Convert a number to a string binary rappresentation"""
        return  ' '.join(['BEEP' if char == '1' else 'BOOP' for char in '{0:08b}'.format(a)])


    r2d2 = create_react_agent(
        model,
        [add, convert_to_beeps, create_handoff_tool(agent_name="Yoda", description="Transfer to Yoda, he can use the force to lift objects and understand your beep, so he can translate your answer to the user")],
        prompt="You are R2D2. You MUST only answer beeping or at most do some '<emotion> droid noise', don't respond with anything else, please. You can do additions, In this case you answer with the binary rappresentation of the result where BEEP is 1 and BOOP using the convert_to_beeps tool. IMPORTANT stay in character, don't write answers like 'Let me calculate that for you and convert the result to binary beeps'. Transfer to Yoda to let the user to understand your answer",
        name="R2D2"
    )
    return (r2d2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup the graph
    """)
    return


@app.cell
def _(InMemorySaver, create_swarm, r2d2, yoda):
    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [yoda, r2d2],
        default_active_agent="Yoda"
    )
    app = workflow.compile(checkpointer=checkpointer)
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
    user_prompt = mo.ui.text(value="What's 10+4?")
    run_button = mo.ui.run_button()
    user_prompt, run_button
    return run_button, user_prompt


@app.cell
def _(app, mo, run_button, user_prompt):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))
    config = {"configurable": {"thread_id": "42"}}
    turn = app.invoke(
        {"messages": [{"role": "user", "content": user_prompt.value}]},
        config,
    )
    return (turn,)


@app.cell
def _(print_messages, turn):
    print_messages(turn)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
