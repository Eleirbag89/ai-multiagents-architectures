import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple Graph Example
    This is a simple example of a LangGraph that demonstrates how to create a graph with conditional edges and how to invoke it with different inputs.

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
    from typing import TypedDict
    return END, START, StateGraph, TypedDict


@app.cell
def _(mo):
    mo.md(r"""
    ## Define the State
    """)
    return


@app.cell
def _(TypedDict):
    class State(TypedDict, total=False):
        name_input: str
        text: str
    return (State,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the greeting node
    """)
    return


@app.cell
def _(State):
    # Function to create the greeting
    # Inputs is the input of the compiled graph invoke, because it is the first node
    def greeting_node(state: State):
        name_input = state.get("name_input")

        greeting = f"Hello, {name_input}! Welcome to LangGraph."
        return {"text": greeting}
    return (greeting_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the emoji node
    """)
    return


@app.cell
def _(State):
    # Function to append an emoji to the text
    def emoji_node(state: State):
        text = state.get("text", "")
        return {"text": text + " 🚀"}
    return (emoji_node,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the conditional edge to choose the next node to execute
    If the input contains "_Gabriele_" we execute the _emoji_ node next, otherwise we terminate the graph execution in the _END_ node
    """)
    return


@app.cell
def _(State):
    # Function to determine the next node based on the input
    def next_node_after_greeting(state: State):
        # If the input contains "Gabriele", go to the "emoji" node
        if "Gabriele" in state.get("text", ""):
            return "if name is Gabriele"
        # Otherwise, go to the END node
        # END is a special node that indicates the end of the graph. If it's missing, the graph will not return anything.
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
    graph = StateGraph(State)

    graph.add_node("greeting", greeting_node)
    graph.add_node("emoji", emoji_node)

    graph.add_edge(START, "greeting")
    # or graph.set_entry_point("greeting")
    graph.add_edge("emoji", END) 
    # or graph.set_finish_point("emoji")

    # Add a conditional edge from the "greeting" node to the node name returned by the function an mapped by path_map
    graph.add_conditional_edges("greeting", next_node_after_greeting, path_map={
        "if name is Gabriele": "emoji",
        "else": END
    })
    return (graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compiling the graph
    Before we can execute a graph, we need to compile it
    """)
    return


@app.cell
def _(graph):
    compiled_graph = graph.compile()
    return (compiled_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the graph
    """)
    return


@app.cell
def _(compiled_graph, mo):
    mo.mermaid(compiled_graph.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's test it
    """)
    return


@app.cell
def _(compiled_graph):
    tests = ["Mario", "Pippo", "Gabriele", "Riccardo"]

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
