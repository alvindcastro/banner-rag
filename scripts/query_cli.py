"""
scripts/query_cli.py
Interactive CLI to query the Banner RAG system directly (no API server needed).

Usage:
    python scripts/query_cli.py
    python scripts/query_cli.py --question "What changed in Banner Finance 9.3.22?"
    python scripts/query_cli.py --question "..." --module Finance --version 9.3.22
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from app.models import AskRequest
from app.rag import ask

console = Console()


def print_response(response):
    # Answer panel
    console.print()
    console.print(Panel(
        response.answer,
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Sources table
    if response.sources:
        table = Table(title="Sources Retrieved", box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("File", style="cyan")
        table.add_column("Page", justify="right", width=5)
        table.add_column("Module", width=12)
        table.add_column("Version", width=10)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Excerpt", max_width=50)

        for i, src in enumerate(response.sources, 1):
            table.add_row(
                str(i),
                src.filename,
                str(src.page or "-"),
                src.banner_module or "-",
                src.banner_version or "-",
                f"{src.score:.3f}",
                src.chunk_text[:120] + "..." if len(src.chunk_text) > 120 else src.chunk_text,
            )
        console.print(table)
    else:
        console.print("[yellow]No source chunks retrieved.[/yellow]")


def interactive_mode():
    console.rule("[bold blue]Ellucian Banner Upgrade Assistant[/bold blue]")
    console.print("Type your question and press Enter. Type [bold]exit[/bold] to quit.\n")

    while True:
        try:
            question = console.input("[bold cyan]Question:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if not question:
            continue

        with console.status("[bold green]Searching knowledge base...[/bold green]"):
            try:
                response = ask(AskRequest(question=question))
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                continue

        print_response(response)
        console.print()


def single_query(question: str, module: str = None, version: str = None, top_k: int = 5):
    console.rule("[bold blue]Banner RAG Query[/bold blue]")
    console.print(f"[bold]Q:[/bold] {question}")
    if module:
        console.print(f"[bold]Module filter:[/bold] {module}")
    if version:
        console.print(f"[bold]Version filter:[/bold] {version}")

    with console.status("[bold green]Searching...[/bold green]"):
        response = ask(AskRequest(
            question=question,
            top_k=top_k,
            module_filter=module,
            version_filter=version,
        ))

    print_response(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Banner RAG knowledge base")
    parser.add_argument("--question", "-q", help="Question to ask (interactive mode if omitted)")
    parser.add_argument("--module", "-m", help="Filter by Banner module (e.g. Finance)")
    parser.add_argument("--version", "-v", help="Filter by Banner version (e.g. 9.3.22)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.question:
        single_query(args.question, module=args.module, version=args.version, top_k=args.top_k)
    else:
        interactive_mode()
