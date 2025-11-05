import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple Graph LLM Example
    This is a simple example of a LangGraph that demonstrates how to create a graph with conditional edges and how to invoke it with different inputs using LLMs.

    The graph consists of two nodes: "_greeting_" and "_emoji_".

    The "_greeting_" node generates a greeting message based on the input name, and the "_emoji_" node appends an emoji to the text.
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
    from langgraph.graph import StateGraph, START, END
    from langchain_openai import ChatOpenAI
    import json
    from typing import TypedDict, Optional, Any, Dict
    return ChatOpenAI, END, START, StateGraph, TypedDict, json


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
def _(mo):
    mo.md(r"""
    #Define the state
    """)
    return


@app.cell
def _(TypedDict):
    class State(TypedDict, total=False):
        name_input: str
        text: str
        add_emoji: bool
    return (State,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the greeting node
    """)
    return


@app.cell
def _(State, json, model):
    def greeting_node(state: State) -> State:
        name_input = state.get("name_input", "studente")

        prompt = (
            'Rispondi usando questa struttura di JSON: '
            '{ "text": "Messaggio generato", "add_emoji": "boolean" } '
            f'Genera un messaggio di saluto per {name_input} che sta partecipando alla DevFest Catania 2025 al talk Orchestrare l\'intelligenza - esplorando le principali architetture multi-agente per l\'AI'
            "e se si chiama Gabriele, setta il flag 'add_emoji' a true."
        )
        messages = [("human", prompt)]

        response = model.invoke(messages)
        content = getattr(response, "content", None)
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            pass

        return {"text": "", "add_emoji": False}
    return (greeting_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the emoji node
    """)
    return


@app.cell
def _(State, model):
    def emoji_node(state: State) -> State:
        """
        Aggiunge un'emoji al testo presente nello state.
        Restituisce il frammento di stato aggiornato {"text": "<testo+emoji>"}.
        """
        text = state.get("text", "")
        print("Original message:", text)

        prompt = (
            f"Aggiungi una emoji super mega swag con rizz al seguente testo: '{text}'. "
            "IMPORTANTE: rispondi solo con il testo e l'emoji, non aggiungere altro!"
        )
        messages = [("human", prompt)]
        response = model.invoke(messages)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)

        # restituiamo il testo aggiornato (soprascriviamo text)
        return {"text": content, "add_emoji": False}
    return (emoji_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the conditional edge to choose the next node to execute
    If the input contains the attribute "add_emoji" sets to True we execute the _emoji_ node next, otherwise we terminate the graph execution in the _END_ node
    """)
    return


@app.cell
def _(State):
    # Funzione di instradamento dopo il nodo di greeting
    def next_node_after_greeting(state: State) -> str:
        # ritorna la 'path key' che viene mappata a un nodo (path_map sotto)
        if state.get("add_emoji", False):
            return "if add_emoji"
        return "else"
    return (next_node_after_greeting,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's create the graph
    """)
    return


@app.cell
def _(
    END,
    START,
    State,
    StateGraph,
    emoji_node,
    greeting_node,
    next_node_after_greeting,
):
    g_builder = StateGraph(State)


    # aggiungo i nodi
    g_builder.add_node("greeting", greeting_node)
    g_builder.add_node("emoji", emoji_node)

    # punti di ingresso/uscita
    g_builder.add_edge(START, "greeting")
    g_builder.add_edge("emoji", END)

    # edge condizionali: il nodo 'greeting' chiama next_node_after_greeting(state)
    g_builder.add_conditional_edges(
        "greeting",
        next_node_after_greeting,
        path_map={
            "if add_emoji": "emoji",
            "else": END,
        },
    )
    return (g_builder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compiling the graph
    Before we can execute a graph, we need to compile it
    """)
    return


@app.cell
def _(g_builder):
    compiled_graph = g_builder.compile()
    return (compiled_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the graph
    """)
    return


@app.cell(hide_code=True)
def _(compiled_graph, mo):
    mo.mermaid(compiled_graph.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's test it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button()
    run_button
    return (run_button,)


@app.cell
def _(compiled_graph, mo, run_button):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))
    tests = ["Mario", "Gabriele", "Pippo"]
    for user in tests:
        result = compiled_graph.invoke({"name_input": user})
        print(f"Input={user} -> {result.get('text')}")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
