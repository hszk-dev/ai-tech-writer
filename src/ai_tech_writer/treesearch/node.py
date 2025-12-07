"""Tree search node and state definitions."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..models import Article, ArticleIdea, ArticleOutline


class SearchStage(str, Enum):
    """Stages of the tree search process."""

    INITIAL = "initial"  # Idea and outline generation
    EXPANDED = "expanded"  # Section detail expansion
    ENHANCED = "enhanced"  # Code example enhancement
    POLISHED = "polished"  # Final polish


@dataclass
class ArticleState:
    """State of an article at a given search node."""

    idea: Optional[ArticleIdea] = None
    outline: Optional[ArticleOutline] = None
    draft: Optional[Article] = None

    def get_current_content(self) -> Optional[str]:
        """Get the current content representation for evaluation."""
        if self.draft:
            return self.draft.to_markdown()
        if self.outline:
            sections = "\n".join(
                f"- {s.heading}: {', '.join(s.key_points)}" for s in self.outline.sections
            )
            return f"Title: {self.outline.idea.title}\n\nSections:\n{sections}"
        if self.idea:
            return f"Title: {self.idea.title}\nTopics: {', '.join(self.idea.topics)}"
        return None

    def has_draft(self) -> bool:
        """Check if a draft exists."""
        return self.draft is not None

    def has_outline(self) -> bool:
        """Check if an outline exists."""
        return self.outline is not None

    def has_idea(self) -> bool:
        """Check if an idea exists."""
        return self.idea is not None


@dataclass
class TreeNode:
    """A node in the search tree representing an article state."""

    id: str
    stage: SearchStage
    article_state: ArticleState
    score: float = 0.0
    depth: int = 0
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_pruned: bool = False

    @classmethod
    def create_root(cls, topic: str) -> "TreeNode":
        """Create a root node for a topic."""
        return cls(
            id=f"root_{uuid.uuid4().hex[:8]}",
            stage=SearchStage.INITIAL,
            article_state=ArticleState(),
            depth=0,
            metadata={"topic": topic},
        )

    @classmethod
    def create_child(
        cls,
        parent: "TreeNode",
        article_state: ArticleState,
        stage: SearchStage,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TreeNode":
        """Create a child node from a parent."""
        node = cls(
            id=f"node_{uuid.uuid4().hex[:8]}",
            stage=stage,
            article_state=article_state,
            depth=parent.depth + 1,
            parent_id=parent.id,
            metadata=metadata or {},
        )
        return node

    def add_child(self, child_id: str) -> None:
        """Add a child node ID."""
        if child_id not in self.children:
            self.children.append(child_id)

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (fully polished)."""
        return self.stage == SearchStage.POLISHED

    def can_expand(self) -> bool:
        """Check if this node can be expanded further."""
        return not self.is_terminal() and not self.is_pruned

    def get_next_stage(self) -> Optional[SearchStage]:
        """Get the next stage for expansion."""
        stage_order = [
            SearchStage.INITIAL,
            SearchStage.EXPANDED,
            SearchStage.ENHANCED,
            SearchStage.POLISHED,
        ]
        try:
            current_idx = stage_order.index(self.stage)
            if current_idx < len(stage_order) - 1:
                return stage_order[current_idx + 1]
        except ValueError:
            pass
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "stage": self.stage.value,
            "depth": self.depth,
            "score": self.score,
            "children": self.children,
            "is_pruned": self.is_pruned,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "article_state": {
                "has_idea": self.article_state.has_idea(),
                "has_outline": self.article_state.has_outline(),
                "has_draft": self.article_state.has_draft(),
            },
        }


@dataclass
class SearchProgress:
    """Progress state for search visualization."""

    iteration: int = 0
    max_iterations: int = 10
    total_nodes: int = 0
    evaluated_nodes: int = 0
    best_score: float = 0.0
    best_node_id: Optional[str] = None
    current_stage: str = ""
    current_action: str = ""
    elapsed_time: float = 0.0
    start_time: Optional[datetime] = None

    def start(self) -> None:
        """Start tracking time."""
        self.start_time = datetime.now()

    def update_elapsed(self) -> None:
        """Update elapsed time."""
        if self.start_time:
            self.elapsed_time = (datetime.now() - self.start_time).total_seconds()

    def get_progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.max_iterations == 0:
            return 0.0
        return (self.iteration / self.max_iterations) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "total_nodes": self.total_nodes,
            "evaluated_nodes": self.evaluated_nodes,
            "best_score": self.best_score,
            "best_node_id": self.best_node_id,
            "current_stage": self.current_stage,
            "current_action": self.current_action,
            "elapsed_time": self.elapsed_time,
            "progress_percent": self.get_progress_percent(),
        }


@dataclass
class IterationState:
    """State captured at the end of each iteration."""

    iteration: int
    timestamp: datetime
    action: str
    expanded_nodes: list[str]
    new_nodes: list[str]
    evaluations: dict[str, float]
    best_node_id: str
    best_score: float
    total_nodes: int
    pruned_nodes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "expanded_nodes": self.expanded_nodes,
            "new_nodes": self.new_nodes,
            "evaluations": self.evaluations,
            "best_node_id": self.best_node_id,
            "best_score": self.best_score,
            "total_nodes": self.total_nodes,
            "pruned_nodes": self.pruned_nodes,
        }


@dataclass
class SearchResult:
    """Result of a tree search."""

    best_node: TreeNode
    path: list[str]
    final_score: float
    total_iterations: int
    total_nodes_explored: int
    total_time_seconds: float
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "final_score": self.final_score,
            "total_iterations": self.total_iterations,
            "total_nodes_explored": self.total_nodes_explored,
            "total_time_seconds": self.total_time_seconds,
            "selection_reason": self.selection_reason,
        }
