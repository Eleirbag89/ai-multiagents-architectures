import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Register the prompts for the planner in MLFlow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run MlFlow server and configure it
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

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    return avvia_mlflow_server, mlflow


@app.cell
def _(mo):
    start_mlflow_button = mo.ui.run_button(label="Start MLFlow server")
    start_mlflow_button
    return (start_mlflow_button,)


@app.cell
def _(avvia_mlflow_server, start_mlflow_button):
    if start_mlflow_button.value:
        print(avvia_mlflow_server())
    return


@app.cell
def _(mlflow):
    def register_prompts():
        # Use double curly braces for variables in the template
        evaluator_prompt = """\
        Hai eseguito il task: '{{last_task}}' con risultato: '{{last_output}}'. Valuta se il task è andato a buon fine e, in caso negativo, fornisci la reason. Non valutare mai la correttezza delle date"
        """

        # Register evaluator prompt
        prompt = mlflow.genai.register_prompt(
            name="evaluator_prompt",
            template=evaluator_prompt,
            # Optional: Provide a commit message to describe the changes
            commit_message="Initial commit",
            # Optional: Set tags applies to the prompt (across versions)
            tags={
                "task": "planner",
                "language": "it",
            },
        )

        # The prompt object contains information about the registered prompt
        print(f"Created prompt '{prompt.name}' (version {prompt.version})")

        executor_sys_prompt = """
        Sei un esecutore di task con accesso a un tool per effettuare una ricerca sul web e uno per inviare una mail. Se non sai quale sia il destinatario non inviare la mail, limitati a rispondere con il body e il subject
        """

        prompt = mlflow.genai.register_prompt(
            name="executor_sys_prompt",
            template=executor_sys_prompt,
            # Optional: Provide a commit message to describe the changes
            commit_message="Initial commit",
            # Optional: Set tags applies to the prompt (across versions)
            tags={
                "task": "planner",
                "language": "it",
            },
        )

        print(f"Created prompt '{prompt.name}' (version {prompt.version})")

        executor_user_prompt = """
        Questo è il piano attuale con task completati in precedenza con il relativo risultato e futuri:
        {{plan_tasks}}
        Tienine conto solo per contesto e per sapere i risultati delle operazioni precedenti, ma della pianificazione dei prossimi task si occupa un agente dedicato, non tu. 
        Non comportarti come un chatbot, ma come un esecutore di task, riporta i risultati in maniera completa ma breve, non aggiungere domande chiedendo se serve ulteriore aiuto.
        Non tornare mai un messaggio vuoto altrimenti verrai valutato negativamente, indica il risultato del task eseguito
        Stai attento alla differenza fra PREPARARE e INVIARE una mail. Nel primo caso non devi inviarla, solo scriverne il contenuto

        Il tuo compito è di eseguire ESCLUSIVAMENTE con 1 tool e ESCLUSIVAMENTE questo task: '{{task}}'"""

        prompt = mlflow.genai.register_prompt(
            name="executor_user_prompt",
            template=executor_user_prompt,
            # Optional: Provide a commit message to describe the changes
            commit_message="Initial commit",
            # Optional: Set tags applies to the prompt (across versions)
            tags={
                "task": "planner",
                "language": "it",
            },
        )

        print(f"Created prompt '{prompt.name}' (version {prompt.version})")

        plan_prompt = """
        Genera un piano step-by-step per l'obiettivo: '{{user_query}}'.
        Ogni passo deve essere un task da eseguire.
        {{tools_prompt}}
        Non aggiungere task non rilevanti per l'obiettivo o non indicati espressamente (es. login o accessi su sistemi e generazione di report/documentazione) focalizzati sulla richiesta dell'utente.
        Riporta nello step del piano le informazioni fornite dall'utente che possano essere utili alla sua esecuzione con successo, considerando che l'executor non ha accesso alla query dell'utente. In particolare, quando chiedi di inviare una mail, riporta sempre l'indirizzo del destinatario.
        """

        prompt = mlflow.genai.register_prompt(
            name="plan_prompt",
            template=plan_prompt,
            # Optional: Provide a commit message to describe the changes
            commit_message="Initial commit",
            # Optional: Set tags applies to the prompt (across versions)
            tags={
                "task": "planner",
                "language": "it",
            },
        )

        print(f"Created prompt '{prompt.name}' (version {prompt.version})")
    return (register_prompts,)


@app.cell(hide_code=True)
def _(mo):
    register_prompts_button = mo.ui.run_button(label="Register prompts")
    register_prompts_button
    return (register_prompts_button,)


@app.cell(hide_code=True)
def _(register_prompts, register_prompts_button):
    if register_prompts_button.value:
        print(register_prompts())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
