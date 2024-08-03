import typer
from personalities import personalities

from llm_w_mlx.model import LLM

app = typer.Typer()


@app.command()
def start_chat(
    personality: str = typer.Option("dwight", help="Personality for the chat."),
    model: str = typer.Option("Mistral-7B-Instruct-v0.1", help="Model name."),
    weights: str = typer.Option(..., help="Model weights path (npz file)."),
    tokenizer: str = typer.Option(..., help="Model tokenizer path (model file)."),
    max_tokens: int = typer.Option(500, help="Max tokens for the chat."),
) -> None:
    print(f"> LLM with personality: {personality.upper()}")

    llm = LLM.build(
        model_name=model,
        weights_path=weights,
        tokenizer_path=tokenizer,
        personality=personalities[personality]["personality"],
        examples=personalities[personality]["examples"],
    )

    llm.chat(max_tokens=max_tokens)


if __name__ == "__main__":
    app()
