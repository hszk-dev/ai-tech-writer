"""Best-First Tree Search implementation for article generation."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..llm import LLMClient, Message, PromptLoader
from ..llm.schemas import IMPROVED_DRAFT_SCHEMA, ImprovedDraftOutput
from ..models import Article, ArticleIdea, ArticleOutline
from ..pipeline.base import StageContext
from .evaluator import ArticleEvaluator, EvaluationResult
from .node import (
    ArticleState,
    IterationState,
    SearchProgress,
    SearchResult,
    SearchStage,
    TreeNode,
)
from .tree import SearchTree

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for tree search."""

    max_iterations: int = 10
    beam_width: int = 3
    max_depth: int = 4
    score_threshold: float = 8.5
    parallel_workers: int = 3
    rate_limit_per_minute: int = 20
    max_nodes: int = 100
    prune_keep_top: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        """Create from dictionary."""
        return cls(
            max_iterations=data.get("max_iterations", 10),
            beam_width=data.get("beam_width", 3),
            max_depth=data.get("max_depth", 4),
            score_threshold=data.get("score_threshold", 8.5),
            parallel_workers=data.get("parallel_workers", 3),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 20),
            max_nodes=data.get("max_nodes", 100),
            prune_keep_top=data.get("prune_keep_top", 20),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "score_threshold": self.score_threshold,
            "parallel_workers": self.parallel_workers,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "max_nodes": self.max_nodes,
            "prune_keep_top": self.prune_keep_top,
        }


class WorkerPool:
    """Manages concurrent task execution with rate limiting."""

    def __init__(self, max_workers: int = 3, rate_limit_per_minute: int = 20):
        """Initialize the worker pool.

        Args:
            max_workers: Maximum concurrent workers
            rate_limit_per_minute: Maximum API calls per minute
        """
        self.max_workers = max_workers
        self.rate_limit_per_minute = rate_limit_per_minute
        self.semaphore = asyncio.Semaphore(max_workers)
        self._last_call_time: float = 0
        self._call_count: int = 0
        self._lock = asyncio.Lock()

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        async with self._lock:
            current_time = asyncio.get_running_loop().time()
            elapsed = current_time - self._last_call_time

            # Reset count if a minute has passed
            if elapsed >= 60:
                self._call_count = 0
                self._last_call_time = current_time

            # Wait if we've hit the limit
            if self._call_count >= self.rate_limit_per_minute:
                wait_time = 60 - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._call_count = 0
                self._last_call_time = asyncio.get_running_loop().time()

            self._call_count += 1

    async def execute(
        self,
        tasks: list[Coroutine[Any, Any, Any]],
    ) -> list[Any]:
        """Execute tasks concurrently with rate limiting.

        Args:
            tasks: List of coroutines to execute

        Returns:
            List of results in same order as tasks
        """

        async def run_with_limits(task: Coroutine[Any, Any, Any]) -> Any:
            await self._rate_limit()
            async with self.semaphore:
                return await task

        wrapped_tasks = [run_with_limits(task) for task in tasks]
        return await asyncio.gather(*wrapped_tasks, return_exceptions=True)


class SearchProgressDisplay:
    """Real-time progress display for tree search."""

    def __init__(self, console: Console, verbose: bool = False):
        """Initialize the display.

        Args:
            console: Rich console for output
            verbose: Whether to show detailed output
        """
        self.console = console
        self.verbose = verbose
        self.live: Optional[Live] = None
        self.progress_state: SearchProgress = SearchProgress()

    def start(self) -> None:
        """Start the live display."""
        if self.verbose:
            self.live = Live(
                self._build_display(),
                console=self.console,
                refresh_per_second=2,
            )
            self.live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def update(self, state: SearchProgress) -> None:
        """Update the progress display.

        Args:
            state: Current search progress
        """
        self.progress_state = state
        if self.live:
            self.live.update(self._build_display())

    def _build_display(self) -> Panel:
        """Build the display panel."""
        state = self.progress_state

        # Status text
        status_text = Text()
        status_text.append("Status: ", style="bold")
        status_text.append(state.current_action or "Initializing...", style="cyan")

        # Create progress bar
        progress_percent = state.get_progress_percent()
        bar_width = 30
        filled = int(bar_width * progress_percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress_text = Text()
        progress_text.append(f"[{bar}] ", style="green")
        progress_text.append(f"{progress_percent:.0f}%", style="bold")

        # Build info table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold", width=12)
        table.add_column("Value")

        table.add_row("Iteration", f"{state.iteration}/{state.max_iterations}")
        table.add_row("Nodes", str(state.total_nodes))
        table.add_row("Evaluated", str(state.evaluated_nodes))
        table.add_row("Best Score", f"{state.best_score:.1f}/10")
        table.add_row("Stage", state.current_stage or "—")
        table.add_row("Time", f"{state.elapsed_time:.1f}s")

        # Best node info
        if state.best_node_id:
            table.add_row("Best Node", f"#{state.best_node_id[-8:]}")

        # Combine elements using Group
        content = Group(
            status_text,
            Text(""),  # Empty line
            progress_text,
            Text(""),  # Empty line
            table,
        )

        return Panel(
            content,
            title="[bold cyan]Tree Search Progress[/bold cyan]",
            border_style="cyan",
        )

    def print_tree(self, tree: SearchTree, topic: str) -> None:
        """Print ASCII tree visualization.

        Args:
            tree: The search tree
            topic: The search topic
        """
        if self.verbose:
            ascii_tree = tree.to_ascii(topic)
            self.console.print(Panel(ascii_tree, title="Search Tree", border_style="dim"))


class IntermediateOutputHandler:
    """Handles saving intermediate search outputs."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        """Initialize the handler.

        Args:
            output_dir: Base output directory
            enabled: Whether to save outputs
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.treesearch_dir: Optional[Path] = None
        self.iterations_dir: Optional[Path] = None
        self.nodes_dir: Optional[Path] = None
        self.evaluations_dir: Optional[Path] = None
        self.articles_dir: Optional[Path] = None

    def initialize(self, config: SearchConfig) -> None:
        """Initialize output directories and save config.

        Args:
            config: Search configuration
        """
        if not self.enabled:
            return

        self.treesearch_dir = self.output_dir / "treesearch"
        self.treesearch_dir.mkdir(parents=True, exist_ok=True)

        self.iterations_dir = self.treesearch_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)

        self.nodes_dir = self.treesearch_dir / "nodes"
        self.nodes_dir.mkdir(exist_ok=True)

        self.evaluations_dir = self.treesearch_dir / "evaluations"
        self.evaluations_dir.mkdir(exist_ok=True)

        self.articles_dir = self.treesearch_dir / "articles"
        self.articles_dir.mkdir(exist_ok=True)

        # Save config
        self._save_json(self.treesearch_dir / "config.json", config.to_dict())

    def on_iteration_complete(self, iteration: int, state: IterationState) -> None:
        """Save iteration state.

        Args:
            iteration: Iteration number
            state: Iteration state
        """
        if not self.enabled or not self.iterations_dir:
            return

        filename = f"iteration_{iteration:03d}.json"
        self._save_json(self.iterations_dir / filename, state.to_dict())

    def on_node_created(self, node: TreeNode) -> None:
        """Save node data and article content.

        Args:
            node: Created node
        """
        if not self.enabled or not self.nodes_dir:
            return

        # Save node metadata
        filename = f"node_{node.id}.json"
        self._save_json(self.nodes_dir / filename, node.to_dict())

        # Save article content as markdown
        self._save_article_content(node)

    def on_node_evaluated(self, node_id: str, result: EvaluationResult) -> None:
        """Save evaluation result.

        Args:
            node_id: Node ID
            result: Evaluation result
        """
        if not self.enabled or not self.evaluations_dir:
            return

        filename = f"eval_{node_id}.json"
        data = result.to_dict()
        data["evaluated_at"] = datetime.now().isoformat()
        self._save_json(self.evaluations_dir / filename, data)

    def on_search_complete(self, result: SearchResult) -> None:
        """Save final search result.

        Args:
            result: Search result
        """
        if not self.enabled or not self.treesearch_dir:
            return

        self._save_json(self.treesearch_dir / "best_path.json", result.to_dict())

    def save_tree_structure(self, tree: SearchTree, topic: str = "") -> None:
        """Save tree structure and visualization.

        Args:
            tree: Search tree
            topic: Search topic
        """
        if not self.enabled or not self.treesearch_dir:
            return

        # Save JSON structure
        self._save_json(self.treesearch_dir / "tree_structure.json", tree.to_dict())

        # Save ASCII visualization
        ascii_tree = tree.to_ascii(topic)
        vis_path = self.treesearch_dir / "tree_visualization.txt"
        vis_path.write_text(ascii_tree, encoding="utf-8")

    def save_progress(self, progress: SearchProgress) -> None:
        """Save progress state.

        Args:
            progress: Current progress
        """
        if not self.enabled or not self.treesearch_dir:
            return

        self._save_json(self.treesearch_dir / "progress.json", progress.to_dict())

    def _save_json(self, path: Path, data: dict[str, Any]) -> None:
        """Save data as JSON file.

        Args:
            path: File path
            data: Data to save
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save JSON to {path}: {e}")

    def _save_article_content(self, node: TreeNode) -> None:
        """Save article content as markdown file.

        Args:
            node: Node containing article state
        """
        if not self.articles_dir:
            return

        article_state = node.article_state
        content_parts = []

        # Add header with node info
        content_parts.append(f"<!-- Node: {node.id} | Stage: {node.stage.value} -->")
        content_parts.append("")

        # Add idea info if available
        if article_state.idea:
            idea = article_state.idea
            content_parts.append(f"# {idea.title}")
            content_parts.append("")
            if idea.emoji:
                content_parts.append(f"Emoji: {idea.emoji}")
            if idea.topics:
                content_parts.append(f"Topics: {', '.join(idea.topics)}")
            if idea.target_audience:
                content_parts.append(f"Target: {idea.target_audience}")
            content_parts.append("")

        # Add outline if available (and no draft yet)
        if article_state.outline and not article_state.draft:
            outline = article_state.outline
            content_parts.append("## Outline")
            content_parts.append("")
            if outline.introduction:
                content_parts.append(f"**Introduction:** {outline.introduction[:200]}...")
                content_parts.append("")
            for section in outline.sections:
                content_parts.append(f"- **{section.heading}**")
                for point in section.key_points[:3]:
                    content_parts.append(f"  - {point}")
            content_parts.append("")

        # Add full draft if available
        if article_state.draft:
            draft = article_state.draft
            # Use the draft's markdown representation
            content_parts.append(draft.to_markdown())

        # Write to file
        if content_parts:
            try:
                filename = f"{node.id}.md"
                filepath = self.articles_dir / filename
                filepath.write_text("\n".join(content_parts), encoding="utf-8")
            except Exception as e:
                logger.debug(f"Failed to save article content for node {node.id}: {e}")


class BFTSearch:
    """Best-First Tree Search for article generation."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        evaluator: ArticleEvaluator,
        config: SearchConfig,
        console: Optional[Console] = None,
        verbose: bool = False,
    ):
        """Initialize the search.

        Args:
            llm_client: LLM client for generation
            prompt_loader: Prompt template loader
            evaluator: Article evaluator
            config: Search configuration
            console: Rich console for output
            verbose: Whether to show verbose output
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.evaluator = evaluator
        self.config = config
        self.console = console or Console()
        self.verbose = verbose

        self.tree = SearchTree(max_nodes=config.max_nodes)
        self.worker_pool = WorkerPool(
            max_workers=config.parallel_workers,
            rate_limit_per_minute=config.rate_limit_per_minute,
        )
        self.progress = SearchProgress(max_iterations=config.max_iterations)
        self.display = SearchProgressDisplay(self.console, verbose)
        self.output_handler: Optional[IntermediateOutputHandler] = None

    async def search(
        self,
        topic: str,
        context: StageContext,
        output_dir: Optional[Path] = None,
    ) -> Article:
        """Execute tree search to generate the best article.

        Args:
            topic: Article topic
            context: Pipeline context
            output_dir: Optional output directory for intermediate files

        Returns:
            Best generated article
        """
        # Initialize output handler
        if output_dir:
            self.output_handler = IntermediateOutputHandler(output_dir, enabled=True)
            self.output_handler.initialize(self.config)

        # Start progress display
        self.progress.start()
        self.display.start()

        try:
            # Step 1: Generate initial ideas (root nodes)
            self._update_progress("Generating initial ideas...", SearchStage.INITIAL.value)
            root_nodes = await self._generate_initial_ideas(topic, context)

            for node in root_nodes:
                self.tree.add_node(node)
                if self.output_handler:
                    self.output_handler.on_node_created(node)

            if self.verbose:
                self.console.print(
                    f"[dim]Generated {len(root_nodes)} initial nodes, starting evaluation...[/dim]"
                )

            # Evaluate initial nodes
            self._update_progress("Evaluating initial ideas...", SearchStage.INITIAL.value)
            initial_evals = await self._evaluate_nodes(root_nodes)
            for node, eval_result in zip(root_nodes, initial_evals):
                node.score = eval_result.overall_score
                if self.output_handler:
                    self.output_handler.on_node_evaluated(node.id, eval_result)

            # Update best score
            best_node = self.tree.get_best_node()
            if best_node:
                self.progress.best_score = best_node.score
                self.progress.best_node_id = best_node.id

            self.progress.total_nodes = len(self.tree.nodes)
            self.progress.evaluated_nodes = len(root_nodes)

            if self.verbose:
                score = self.progress.best_score
                self.console.print(f"[dim]Initial evaluation done. Best score: {score:.1f}[/dim]")

            # Check if we should skip search loop
            if self.config.max_iterations == 0:
                if self.verbose:
                    self.console.print("[dim]Skipping search loop (max_iterations=0)[/dim]")

            # Step 2: Main search loop
            selection_reason = "Max iterations reached"
            for iteration in range(self.config.max_iterations):
                self.progress.iteration = iteration + 1
                self._update_progress(f"Iteration {iteration + 1}...", "")

                if self.verbose:
                    self.console.print(f"[dim]Starting iteration {iteration + 1}...[/dim]")

                # Get best expandable nodes
                expandable = self.tree.get_expandable_nodes()
                if self.verbose:
                    self.console.print(f"[dim]Found {len(expandable)} expandable nodes[/dim]")

                if not expandable:
                    if self.verbose:
                        self.console.print("[dim]No expandable nodes, stopping search[/dim]")
                    break

                best_to_expand = sorted(
                    expandable,
                    key=lambda n: n.score,
                    reverse=True,
                )[: self.config.beam_width]

                if self.verbose:
                    for n in best_to_expand:
                        self.console.print(
                            f"[dim]  Node {n.id[-8:]} depth={n.depth} stage={n.stage.value}[/dim]"
                        )

                # Expand selected nodes
                new_nodes = []
                expanded_ids = []
                for node in best_to_expand:
                    if node.depth >= self.config.max_depth:
                        if self.verbose:
                            self.console.print(
                                f"[dim]  Skipping node {node.id[-8:]} (max depth reached)[/dim]"
                            )
                        continue

                    self._update_progress(
                        f"Expanding node {node.id[-4:]}...",
                        node.stage.value,
                    )

                    if self.verbose:
                        self.console.print(f"[dim]  Expanding node {node.id[-8:]}...[/dim]")

                    children = await self._expand_node(node, context)

                    if self.verbose:
                        self.console.print(f"[dim]  Generated {len(children)} children[/dim]")

                    new_nodes.extend(children)
                    expanded_ids.append(node.id)

                # Add new nodes to tree
                for node in new_nodes:
                    self.tree.add_node(node)
                    if self.output_handler:
                        self.output_handler.on_node_created(node)

                # Evaluate new nodes
                self._update_progress("Evaluating nodes...", "")
                evaluations = await self._evaluate_nodes(new_nodes)

                # Update scores
                eval_dict = {}
                for node, result in zip(new_nodes, evaluations):
                    node.score = result.overall_score
                    eval_dict[node.id] = result.overall_score
                    if self.output_handler:
                        self.output_handler.on_node_evaluated(node.id, result)

                # Prune if needed
                pruned = self.tree.prune(self.config.prune_keep_top)

                # Update progress
                best_node = self.tree.get_best_node()
                if best_node:
                    self.progress.best_score = best_node.score
                    self.progress.best_node_id = best_node.id

                self.progress.total_nodes = len(self.tree.nodes)
                evaluated = [n for n in self.tree.nodes.values() if n.score > 0]
                self.progress.evaluated_nodes = len(evaluated)

                # Save iteration state
                if self.output_handler:
                    iter_state = IterationState(
                        iteration=iteration + 1,
                        timestamp=datetime.now(),
                        action="expand",
                        expanded_nodes=expanded_ids,
                        new_nodes=[n.id for n in new_nodes],
                        evaluations=eval_dict,
                        best_node_id=best_node.id if best_node else "",
                        best_score=self.progress.best_score,
                        total_nodes=self.progress.total_nodes,
                        pruned_nodes=pruned,
                    )
                    self.output_handler.on_iteration_complete(iteration + 1, iter_state)

                # Check termination conditions
                if best_node and best_node.score >= self.config.score_threshold:
                    selection_reason = f"Score threshold ({self.config.score_threshold}) reached"
                    break

            # Get final result
            best_node = self.tree.get_best_node()
            if not best_node or not best_node.article_state.draft:
                raise RuntimeError("No valid article generated")

            # Calculate elapsed time
            self.progress.update_elapsed()

            # Save final outputs
            if self.output_handler:
                search_result = SearchResult(
                    best_node=best_node,
                    path=self.tree.get_path_to_node(best_node.id),
                    final_score=best_node.score,
                    total_iterations=self.progress.iteration,
                    total_nodes_explored=self.progress.total_nodes,
                    total_time_seconds=self.progress.elapsed_time,
                    selection_reason=selection_reason,
                )
                self.output_handler.on_search_complete(search_result)
                self.output_handler.save_tree_structure(self.tree, topic)
                self.output_handler.save_progress(self.progress)

            # Show final tree
            self.display.print_tree(self.tree, topic)

            return best_node.article_state.draft

        finally:
            self.display.stop()

    def _update_progress(self, action: str, stage: str) -> None:
        """Update progress state and display.

        Args:
            action: Current action description
            stage: Current stage
        """
        self.progress.current_action = action
        self.progress.current_stage = stage
        self.progress.update_elapsed()
        self.display.update(self.progress)

    async def _generate_initial_ideas(
        self,
        topic: str,
        context: StageContext,
    ) -> list[TreeNode]:
        """Generate initial idea nodes.

        Args:
            topic: Article topic
            context: Pipeline context

        Returns:
            List of initial nodes with ideas
        """
        # Generate multiple ideas with different approaches
        approaches = ["tutorial", "deep-dive", "practical"]
        tasks = []

        for approach in approaches[: self.config.beam_width]:
            tasks.append(self._generate_idea(topic, approach, context))

        results = await self.worker_pool.execute(tasks)

        nodes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log the error for debugging
                approach = approaches[i] if i < len(approaches) else "default"
                self.console.print(
                    f"[yellow]Warning: Approach '{approach}' failed: {result}[/yellow]"
                )
                continue

            idea, outline, draft = result
            state = ArticleState(idea=idea, outline=outline, draft=draft)
            node = TreeNode.create_root(topic)
            node.article_state = state
            node.stage = SearchStage.EXPANDED if draft else SearchStage.INITIAL
            node.metadata["approach"] = approaches[i] if i < len(approaches) else "default"
            nodes.append(node)

        if not nodes:
            raise RuntimeError(
                "Failed to generate any initial ideas. "
                "Check API keys (ANTHROPIC_API_KEY, TAVILY_API_KEY) and network connection."
            )

        return nodes

    async def _generate_idea(
        self,
        topic: str,
        approach: str,
        context: StageContext,
    ) -> tuple[ArticleIdea, Optional[ArticleOutline], Optional[Article]]:
        """Generate a single idea with optional outline and draft.

        Args:
            topic: Article topic
            approach: Generation approach
            context: Pipeline context

        Returns:
            Tuple of (idea, outline, draft)
        """
        # Import stages locally to avoid circular imports
        from ..pipeline.draft import DraftStage
        from ..pipeline.ideation import IdeationStage
        from ..pipeline.outline import OutlineStage

        if self.verbose:
            self.console.print(f"[dim]  [{approach}] Starting ideation...[/dim]")

        # Run ideation
        ideation = IdeationStage()
        idea = await ideation.execute({"topic": topic, "approach": approach}, context)

        if self.verbose:
            self.console.print(f"[dim]  [{approach}] Ideation done, starting outline...[/dim]")

        # Run outline
        outline_stage = OutlineStage()
        outline = await outline_stage.execute(idea, context)

        if self.verbose:
            self.console.print(f"[dim]  [{approach}] Outline done, starting draft...[/dim]")

        # Run draft
        draft_stage = DraftStage()
        draft = await draft_stage.execute(outline, context)

        if self.verbose:
            self.console.print(f"[dim]  [{approach}] Draft done![/dim]")

        return idea, outline, draft

    async def _expand_node(
        self,
        node: TreeNode,
        context: StageContext,
    ) -> list[TreeNode]:
        """Expand a node to create children.

        Args:
            node: Node to expand
            context: Pipeline context

        Returns:
            List of child nodes
        """
        next_stage = node.get_next_stage()
        if not next_stage:
            return []

        # Generate variants based on current stage
        variants = await self._generate_variants(node, context, num_variants=2)

        children = []
        for variant_state in variants:
            child = TreeNode.create_child(
                parent=node,
                article_state=variant_state,
                stage=next_stage,
            )
            children.append(child)

        return children

    async def _generate_variants(
        self,
        node: TreeNode,
        context: StageContext,
        num_variants: int = 2,
    ) -> list[ArticleState]:
        """Generate variant article states from a node.

        Args:
            node: Source node
            context: Pipeline context
            num_variants: Number of variants to generate

        Returns:
            List of variant ArticleStates
        """
        variants: list[ArticleState] = []
        current_state = node.article_state

        if not current_state.draft:
            return variants

        # Generate improved versions with different temperatures
        temperatures = [0.5, 0.8, 1.0][:num_variants]

        for temp in temperatures:
            if self.verbose:
                self.console.print(f"[dim]    Improving draft with temp={temp}...[/dim]")

            improved_draft = await self._improve_draft(
                current_state.draft,
                context,
                temperature=temp,
            )
            if improved_draft:
                new_state = ArticleState(
                    idea=current_state.idea,
                    outline=current_state.outline,
                    draft=improved_draft,
                )
                variants.append(new_state)

                if self.verbose:
                    self.console.print(f"[dim]    Variant created (temp={temp})[/dim]")

        return variants

    async def _improve_draft(
        self,
        draft: Article,
        context: StageContext,
        temperature: float = 0.7,
    ) -> Optional[Article]:
        """Improve an existing draft.

        Args:
            draft: Current draft
            context: Pipeline context
            temperature: Generation temperature

        Returns:
            Improved draft or None
        """
        # Get current content
        content = draft.to_markdown()

        prompt = f"""以下の技術記事を改善してください。

## 現在の記事
{content}

## 改善指示
- コード例をより実践的にする
- 説明をより分かりやすくする
- 構成を見直す

## 出力形式
改善した記事のセクション内容をJSON形式で出力してください：
```json
{{
  "sections": [
    {{"heading": "セクション名", "content": "内容..."}}
  ]
}}
```
"""

        messages = [
            Message(
                role="system",
                content="あなたは技術記事の編集者です。記事の品質を向上させてください。",
            ),
            Message(role="user", content=prompt),
        ]

        try:
            # Temporarily adjust temperature
            original_temp = context.config.get("llm", {}).get("temperature", 0.7)
            context.config.setdefault("llm", {})["temperature"] = temperature

            result = await context.llm_client.complete_json(messages, schema=IMPROVED_DRAFT_SCHEMA)
            validated = ImprovedDraftOutput.model_validate(result)

            # Restore temperature
            context.config["llm"]["temperature"] = original_temp

            # Build improved article from validated output
            from ..models import ArticleSection

            improved_sections = [
                ArticleSection(
                    heading=section.heading,
                    level=2,
                    content=section.content,
                )
                for section in validated.sections
            ]

            if not improved_sections:
                return draft  # Return original if improvement failed

            return Article(
                frontmatter=draft.frontmatter,
                sections=improved_sections,
                references=draft.references,
            )

        except Exception as e:
            logger.debug(f"Failed to improve draft: {e}")
            return draft  # Return original on error

    async def _evaluate_nodes(
        self,
        nodes: list[TreeNode],
    ) -> list[EvaluationResult]:
        """Evaluate multiple nodes.

        Args:
            nodes: Nodes to evaluate

        Returns:
            List of evaluation results
        """
        states_with_ids = [(node.article_state, node.id) for node in nodes]

        return await self.evaluator.evaluate_batch(
            states_with_ids,
            max_concurrent=self.config.parallel_workers,
        )
