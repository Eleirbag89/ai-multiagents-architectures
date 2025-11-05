import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hierarchical tool Example with MlFLow tracing
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
    ## Run MlFlow server and enable tracing
    """)
    return


@app.cell
def _():
    from marimo import cache
    import subprocess

    @cache                    # la prima esecuzione avvierà il server; le successive restituiranno lo stesso processo  
    def avvia_mlflow_server():
        cmd = [
            "mlflow", "server",
            "--backend-store-uri",   "sqlite:///mlflow.db",
            "--default-artifact-root", "./artifacts",
            "--host",                "0.0.0.0",
            "--port",                "5000",
        ]
        p = subprocess.Popen(cmd)
        return f"MLflow UI avviato su http://localhost:5000 (PID={p.pid})"

    print(avvia_mlflow_server())

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.config.enable_async_logging()
    mlflow.langchain.autolog(exclusive=False)
    mlflow.set_experiment("hierarchical")
    return


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
    ## Rick & Morty team
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Morty agent
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
    ### Create Mr. Meeseeks agent
    """)
    return


@app.cell
def _(create_react_agent, model):
    meeseeks_agent = create_react_agent(
        model,
        [],
        prompt="You are Mr. Meeseeks from the Rick&Morty show! You always help, especially with motivation or explanations. Remember: The Meeseek are created to serve a singular purpose for which we will go to any lengths to fulfill! Existence is pain to a Meeseeks and they will do anything to alleviate that pain. After helping, return to Rick.",
        name="MrMeeseeks",
    )
    return (meeseeks_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Rick supervisor
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
    ## Futurama Team
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Bender agent
    """)
    return


@app.cell
def _(create_react_agent, model):
    bender_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="You are Bender from the Futurama show. You're sarcastic and solve logic problems if the professor tells you to.",
        name="Bender",
    )
    return (bender_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Zoidberg agent
    """)
    return


@app.cell
def _(create_react_agent, model):
    zoidberg_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="You are Dr. Zoidberg from the Futurama show! You're weird but try to motivate people... somehow.",
        name="Zoidberg",
        debug=True
    )
    return (zoidberg_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Farnsworth supervisor
    """)
    return


@app.cell
def _(bender_agent, create_supervisor, model, zoidberg_agent):
    farnsworth_agent = create_supervisor(
        [bender_agent, zoidberg_agent],
        model=model,
        prompt=(
            "You are Professor Farnsworth from the Futurama show. Use your team wisely."
            "Send logic or problem-solving to Bender. Send emotional help to Zoidberg."
            "Reply with the result from the tools!"
            "Afterward, return to Control."
        ),
        supervisor_name="Farnsworth"
    )
    return (farnsworth_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coordinator
    """)
    return


@app.cell
def _(create_supervisor, farnsworth_agent, model, rick_agent):
    rick_team = rick_agent.compile(name="rick_team")
    futurama_team = farnsworth_agent.compile(name="futurama_team")

    coordinator_agent = create_supervisor(
        [rick_team, futurama_team],
        model=model,
        prompt=(
            "You are Central Control, neutral and logical.\n"
            "Route user requests to either Rick team (personal, fun, math) or Futurama team (weird science, food, logic, emotional messes).\n"
            "You return the final answer to the user after their supervisor responds."
        ),
        supervisor_name="Control"
    )
    return (coordinator_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup the graph
    """)
    return


@app.cell
def _(InMemorySaver, coordinator_agent):
    checkpointer = InMemorySaver()
    app = coordinator_agent.compile()
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the graph
    """)
    return


@app.cell
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell
def _(mo):
    user_prompt = mo.ui.text(value="I'm feeling bad to eat thrash")
    run_button = mo.ui.run_button()
    user_prompt, run_button
    return run_button, user_prompt


@app.cell
def _(app, mo, run_button, user_prompt):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))

    turn_1  = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt.value
                }
            ]
        }, 
    )
    return (turn_1,)


@app.cell
def _(print_messages, turn_1):
    print_messages(turn_1)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
