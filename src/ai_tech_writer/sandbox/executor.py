"""Code execution in Docker sandbox."""

import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .languages import get_language_config


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str
    error: str
    exit_code: int
    timed_out: bool = False


class DockerSandbox:
    """Execute code in Docker containers for safety."""

    def __init__(
        self,
        timeout: int = 30,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
    ):
        """Initialize Docker sandbox.

        Args:
            timeout: Execution timeout in seconds
            memory_limit: Memory limit (e.g., "256m")
            cpu_limit: CPU limit (0.5 = 50% of one core)
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

    def is_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def execute(
        self,
        code: str,
        language: str,
        stdin: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute code in a Docker container.

        Args:
            code: Code to execute
            language: Programming language
            stdin: Optional input to provide to the program

        Returns:
            ExecutionResult with output and status
        """
        config = get_language_config(language)
        if config is None or config.docker_image is None:
            return ExecutionResult(
                success=False,
                output="",
                error=f"No Docker image configured for {language}",
                exit_code=-1,
            )

        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write code to file
            ext = config.extensions[0] if config.extensions else ".txt"
            code_file = tmpdir_path / f"code{ext}"
            code_file.write_text(code, encoding="utf-8")

            # Build Docker command
            container_name = f"sandbox-{uuid.uuid4().hex[:8]}"
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--memory",
                self.memory_limit,
                f"--cpus={self.cpu_limit}",
                "--network",
                "none",  # No network access
                "--read-only",  # Read-only filesystem
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{tmpdir}:/code:ro",
                "-w",
                "/code",
                config.docker_image,
            ]

            # Add run command
            if config.run_command:
                run_cmd = config.run_command.format(file=f"code{ext}")
                docker_cmd.extend(["sh", "-c", run_cmd])
            else:
                docker_cmd.extend(["cat", f"code{ext}"])

            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    input=stdin,
                )

                return ExecutionResult(
                    success=result.returncode == 0,
                    output=result.stdout[:10000],  # Limit output size
                    error=result.stderr[:10000],
                    exit_code=result.returncode,
                )

            except subprocess.TimeoutExpired:
                # Kill the container
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True,
                )
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout}s",
                    exit_code=-1,
                    timed_out=True,
                )

            except Exception as e:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    exit_code=-1,
                )


class LocalExecutor:
    """Execute code locally (less safe, for trusted code only)."""

    def __init__(self, timeout: int = 10):
        """Initialize local executor.

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout

    def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code locally.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output and status
        """
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout[:5000],
                error=result.stderr[:5000],
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout}s",
                exit_code=-1,
                timed_out=True,
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                output="",
                error="Python not found",
                exit_code=-1,
            )

    def execute_node(self, code: str) -> ExecutionResult:
        """Execute JavaScript code locally using Node.js.

        Args:
            code: JavaScript code to execute

        Returns:
            ExecutionResult with output and status
        """
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout[:5000],
                error=result.stderr[:5000],
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout}s",
                exit_code=-1,
                timed_out=True,
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                output="",
                error="Node.js not found",
                exit_code=-1,
            )
