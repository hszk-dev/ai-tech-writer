"""Search tree management."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .node import SearchStage, TreeNode


@dataclass
class SearchTree:
    """Manages the search tree structure."""

    nodes: dict[str, TreeNode] = field(default_factory=dict)
    root_id: Optional[str] = None
    max_nodes: int = 100

    def add_node(self, node: TreeNode) -> None:
        """Add a node to the tree.

        Args:
            node: Node to add
        """
        self.nodes[node.id] = node

        # Set root if this is the first node
        if self.root_id is None:
            self.root_id = node.id

        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].add_child(node.id)

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get a node by ID.

        Args:
            node_id: Node ID to look up

        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_root(self) -> Optional[TreeNode]:
        """Get the root node.

        Returns:
            The root node if exists
        """
        if self.root_id:
            return self.nodes.get(self.root_id)
        return None

    def get_best_nodes(self, n: int = 5) -> list[TreeNode]:
        """Get the top n nodes by score.

        Args:
            n: Number of nodes to return

        Returns:
            List of top nodes sorted by score descending
        """
        non_pruned = [node for node in self.nodes.values() if not node.is_pruned]
        sorted_nodes = sorted(non_pruned, key=lambda x: x.score, reverse=True)
        return sorted_nodes[:n]

    def get_best_node(self) -> Optional[TreeNode]:
        """Get the single best node by score.

        Returns:
            The highest scoring node
        """
        best_nodes = self.get_best_nodes(1)
        return best_nodes[0] if best_nodes else None

    def get_expandable_nodes(self) -> list[TreeNode]:
        """Get nodes that can be expanded.

        Returns:
            List of nodes that can be expanded
        """
        return [node for node in self.nodes.values() if node.can_expand()]

    def get_nodes_by_stage(self, stage: SearchStage) -> list[TreeNode]:
        """Get all nodes at a specific stage.

        Args:
            stage: The stage to filter by

        Returns:
            List of nodes at the given stage
        """
        return [node for node in self.nodes.values() if node.stage == stage and not node.is_pruned]

    def get_nodes_by_depth(self, depth: int) -> list[TreeNode]:
        """Get all nodes at a specific depth.

        Args:
            depth: The depth to filter by

        Returns:
            List of nodes at the given depth
        """
        return [node for node in self.nodes.values() if node.depth == depth and not node.is_pruned]

    def get_path_to_node(self, node_id: str) -> list[str]:
        """Get the path from root to a specific node.

        Args:
            node_id: Target node ID

        Returns:
            List of node IDs from root to target
        """
        path: list[str] = []
        current_id: Optional[str] = node_id

        while current_id:
            path.append(current_id)
            node = self.nodes.get(current_id)
            if node:
                current_id = node.parent_id
            else:
                break

        return list(reversed(path))

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf nodes
        """
        return [node for node in self.nodes.values() if not node.children and not node.is_pruned]

    def prune(self, keep_top: int = 20) -> list[str]:
        """Prune low-scoring nodes, keeping only the top ones.

        Args:
            keep_top: Number of top nodes to keep

        Returns:
            List of pruned node IDs
        """
        if len(self.nodes) <= keep_top:
            return []

        # Get all non-pruned nodes sorted by score
        non_pruned = [node for node in self.nodes.values() if not node.is_pruned]
        sorted_nodes = sorted(non_pruned, key=lambda x: x.score, reverse=True)

        # Mark nodes beyond keep_top as pruned
        pruned_ids = []
        for node in sorted_nodes[keep_top:]:
            # Don't prune if this node is an ancestor of a top node
            if not self._is_ancestor_of_top_nodes(node.id, sorted_nodes[:keep_top]):
                node.is_pruned = True
                pruned_ids.append(node.id)

        return pruned_ids

    def _is_ancestor_of_top_nodes(
        self,
        node_id: str,
        top_nodes: list[TreeNode],
    ) -> bool:
        """Check if a node is an ancestor of any top nodes.

        Args:
            node_id: Node ID to check
            top_nodes: List of top nodes

        Returns:
            True if node is an ancestor of any top node
        """
        for top_node in top_nodes:
            path = self.get_path_to_node(top_node.id)
            if node_id in path:
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get tree statistics.

        Returns:
            Dictionary of tree statistics
        """
        non_pruned = [n for n in self.nodes.values() if not n.is_pruned]
        scores = [n.score for n in non_pruned if n.score > 0]

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(non_pruned),
            "pruned_nodes": len(self.nodes) - len(non_pruned),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores, default=0),
            "min_score": min(scores, default=0),
            "nodes_by_stage": {
                stage.value: len(self.get_nodes_by_stage(stage)) for stage in SearchStage
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert tree to dictionary for serialization.

        Returns:
            Dictionary representation of the tree
        """
        return {
            "root_id": self.root_id,
            "max_nodes": self.max_nodes,
            "statistics": self.get_statistics(),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    def to_ascii(self, topic: str = "") -> str:
        """Generate ASCII visualization of the tree.

        Args:
            topic: Optional topic string for root label

        Returns:
            ASCII art representation of the tree
        """
        if not self.root_id:
            return "Empty tree"

        lines = []
        root_label = f'Root: "{topic}"' if topic else "Root"
        lines.append(root_label)

        root = self.get_root()
        if root:
            self._build_ascii(root, lines, "", True)

        return "\n".join(lines)

    def _build_ascii(
        self,
        node: TreeNode,
        lines: list[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        """Recursively build ASCII tree representation.

        Args:
            node: Current node
            lines: List of lines to append to
            prefix: Current line prefix
            is_last: Whether this is the last child
        """
        # Skip root in recursion (already printed)
        if node.parent_id is not None:
            connector = "└── " if is_last else "├── "
            status = ""
            if node.is_pruned:
                status = " (pruned)"
            elif node == self.get_best_node():
                status = " ★"

            stage_str = node.stage.value.upper()
            node_info = f"#{node.id[-4:]} [{stage_str}] score: {node.score:.1f}{status}"
            line = f"{prefix}{connector}{node_info}"
            lines.append(line)

        # Process children
        children = [self.nodes[child_id] for child_id in node.children if child_id in self.nodes]

        for i, child in enumerate(children):
            is_child_last = i == len(children) - 1
            if node.parent_id is None:
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            self._build_ascii(child, lines, child_prefix, is_child_last)
