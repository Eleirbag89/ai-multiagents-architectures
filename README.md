# AI Multi-Agent Architectures – Example Notebooks

This repository contains example notebooks for the 2025 course [Agenti Intelligenti e Machine Learning (AiTHO)](https://web.dmi.unict.it/it/corsi/l-31/agenti-intelligenti-e-machine-learning-aitho), focusing on multi-agent AI architectures.

## Tech Stack

- **Python**
- **[Marimo](https://marimo.io/)** – A modern alternative to Jupyter for interactive notebooks  
- **[LangGraph](https://github.com/langchain-ai/langgraph)** – A framework for building AI agent workflows

## AI Models

The examples use **Anthropic** models by default. You're welcome to switch to any model provider of your choice.

## Project Structure

All the example notebooks and code are located in the `examples/` directory.

## Setup Instructions

### 1. Install Poetry

Poetry is the dependency manager used in this project. Follow the [official installation guide](https://python-poetry.org/docs/#installation) to set it up on your system.

### 2. Install Project Dependencies

```bash
poetry install
```

### 3. Setup OpenAI key

Copy the file .env.example as .env and put your key in the OPENAI_API_KEY field

### 4. Launch the Notebooks
Use Marimo to edit and run the notebooks:

```bash
poetry run marimo edit
```
