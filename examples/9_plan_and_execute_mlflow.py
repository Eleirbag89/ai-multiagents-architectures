import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plan and execute architecture Example with MLFlow tracing and prompts
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
    from langgraph.graph import StateGraph, END
    from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun, Tool
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from typing import List, TypedDict, Tuple, Annotated, Optional
    from pydantic import BaseModel, Field
    import operator
    from langchain_core.tools import tool
    return (
        Annotated,
        BaseModel,
        ChatOpenAI,
        DuckDuckGoSearchAPIWrapper,
        DuckDuckGoSearchRun,
        END,
        Field,
        List,
        Optional,
        StateGraph,
        Tool,
        Tuple,
        TypedDict,
        create_react_agent,
        operator,
        tool,
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
    mlflow.set_experiment("planner")
    return (mlflow,)


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
    ## Define the support classes
    - Plan contains a list of steps. It's the output type of the Planner Agent
    - PlanExecState contains the state of the system: user request, executed and next steps and the final answer
    - EvalResult models the reponse of the evaluator agent. it's the output type of the Eval agent
    """)
    return


@app.cell
def _(BaseModel, Field, List):
    class Plan(BaseModel):
        """Genera un piano step-by-step per l'obiettivo fornito."""
        steps: List[str] = Field(description="Lista di passi da eseguire")
    return (Plan,)


@app.cell
def _(Annotated, List, Optional, Tuple, TypedDict, operator):
    class PlanExecState(TypedDict):
        input: str
        plan: List[str]
        past_steps: Annotated[List[Tuple[str, str]], operator.add]
        response: str
        reason: Optional[str]
    return (PlanExecState,)


@app.cell
def _(BaseModel, Optional):
    class EvalResult(BaseModel):
        success: bool
        reason: Optional[str] = None
    return (EvalResult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define planner agent
    """)
    return


@app.cell
def _(Plan, PlanExecState, model):
    planner = model.with_structured_output(Plan, method="function_calling")

    def plan_step(state: PlanExecState) -> dict:
        prompt_input = state["input"]
        plan_obj : Plan = planner.invoke(prompt_input)
        return {"plan": plan_obj.steps, "reason": None}
    return (plan_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Executor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the search tool
    """)
    return


@app.cell
def _(DuckDuckGoSearchAPIWrapper, DuckDuckGoSearchRun, Tool):
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

    # 3. Tool: Web search
    search_tool = Tool(
        name="WebSearch",
        func=search.run,
        description="Use this tool to search the web for up-to-date information."
    )
    return (search_tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define a fake tool to sens emails
    """)
    return


@app.cell
def _(tool):
    @tool
    def send_email(receiver: str, subject: str, body: str) -> int:
        """
        Sends an email to the specified recipient.

        Parameters:
            receiver (str): The email address of the recipient.
            subject (str): The subject line of the email.
            body (str): The main content of the email.

        Returns:
            bool: True success, False failure).
        """
        print(f"@@@ Sending email to {receiver} with subject {subject} and body \n{body}\n---")
        return True
    return (send_email,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define executor agent
    """)
    return


@app.cell
def _(
    PlanExecState,
    create_react_agent,
    mlflow,
    model,
    search_tool,
    send_email,
):
    executor_sys_prompt = mlflow.genai.load_prompt("prompts:/executor_sys_prompt/1")
    executor_agent = create_react_agent(
        model=model,
        tools=[search_tool, send_email],
        prompt= executor_sys_prompt.format()
    ,
        name="executor",
        debug=True,
    )

    executor_user_prompt = mlflow.genai.load_prompt("prompts:/executor_user_prompt/1")
    def exec_step(state: PlanExecState) -> dict:

        task = state["plan"].pop(0)
        history = "\n".join([f"{i+1}. {t} -> {o}" for i, (t, o) in enumerate(state["past_steps"])])
        plan_text = history + f"\n{len(state['past_steps'])+1}. {task} -> TODO"
        prompt = executor_user_prompt.format(plan_tasks=plan_text, task=task)
        res = executor_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        ans = res["messages"][-1].content
        return {"past_steps": [(task, ans)]}
    return exec_step, executor_agent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluator
    """)
    return


@app.cell
def _(EvalResult, PlanExecState, mlflow, model):
    evaluator_prompt = mlflow.load_prompt("prompts:/evaluator_prompt/1")

    def evaluate_task_step(state: PlanExecState) -> dict:
        last_task, last_output = state["past_steps"][-1]
        evaluator = model.with_structured_output(EvalResult, method="function_calling")
        prompt = evaluator_prompt.format(last_task=last_task, last_output=last_output)
        eval_result : EvalResult = evaluator.invoke(prompt)

        return {
            "success": eval_result.success,
            "reason": eval_result.reason,
            "plan": state["plan"],
        }
    return (evaluate_task_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finalize
    """)
    return


@app.cell
def _(PlanExecState):
    def finalize(state: PlanExecState) -> dict:
        lines = [f"{i+1}. {task}: {output}" for i, (task, output) in enumerate(state["past_steps"])]
        final = "\n".join(lines)
        return {"response": final}
    return (finalize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the graph
    """)
    return


@app.cell
def _(
    END,
    PlanExecState,
    StateGraph,
    evaluate_task_step,
    exec_step,
    finalize,
    plan_step,
):
    wf = StateGraph(PlanExecState)

    wf.add_node("planner", lambda s: plan_step(s))
    wf.add_node("executor", lambda s: exec_step(s))
    wf.add_node("task_evaluator", evaluate_task_step)
    wf.add_node("finalize", finalize)

    wf.set_entry_point("planner")
    wf.add_edge("planner", "executor")
    wf.add_edge("executor", "task_evaluator")

    wf.add_conditional_edges(
        "task_evaluator",
        lambda s: (
            "executor" if s.get("success") and s.get("plan") else "finalize"
        ),
        {"executor": "executor", "finalize": "finalize"}
    )

    wf.add_edge("finalize", END)

    app = wf.compile()
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Visualize the graph
    """)
    return


@app.cell
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Last touches
    Let's define an helper function to invoke the planner and execute workflow
    """)
    return


@app.cell
def _(List, Optional, PlanExecState, Tool, app, mlflow):
    def run_agent(user_query: str, executor_agent, tools: Optional[List[Tool]] = None):
        tools_prompt = ""
        if tools:
            tools_prompt = (
                "Il tuo executor può usare i seguenti agenti:\n"
                + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
                + "\n"
                "Li userà in automatico. Non è necessario nominarli esplicitamente.\n"
            )

        plan_prompt = mlflow.genai.load_prompt("prompts:/plan_prompt/1").format(
            user_query=user_query, tools_prompt=tools_prompt
        )

        initial_state: PlanExecState = {
            "input": plan_prompt,
            "plan": [],
            "past_steps": [],
            "response": "",
            "reason": None,
            "failure_count": 0,
        }

        return app.invoke(initial_state)
    return (run_agent,)


@app.cell
def _(mo):
    user_prompt = mo.ui.text_area(value="Invia all'indirizzo email paperino@topolin.ia la popolazione della città con il grattacielo più alto del mondo", full_width=True)
    run_button = mo.ui.run_button()
    user_prompt, run_button
    return run_button, user_prompt


@app.cell
def _(
    executor_agent,
    mo,
    run_agent,
    run_button,
    search_tool,
    send_email,
    user_prompt,
):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))
    query = user_prompt.value
    result = run_agent(query, executor_agent, tools=[search_tool, send_email])
    return (result,)


@app.cell
def _(result):
    print(result['response'])
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
