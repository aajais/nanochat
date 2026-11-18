"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
Inspired by the OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

The code runs in a fresh Python subprocess. What is covered:
- Each execution runs in its own process (killed hard by the parent on timeout)
- A fresh interpreter: no access to the parent process memory, and a scrubbed environment
- Memory limits are enforced via rlimits (256MB by default)
- stdout and stderr are captured, stdin is disabled
- Code runs in a temporary directory that is deleted afterwards
- Destructive functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

What is not covered:
- Not a true security sandbox
- Network access is not blocked (e.g. sockets could be opened)
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- No kernel-level isolation (no seccomp, no containers, no virtualization)

Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import subprocess
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing Python code in a sandbox."""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


# The guard runs in the subprocess before the untrusted code. It applies the
# resource limits and disables destructive functions to protect against
# accidents (a fork bomb, deleting files, killing other processes, ...).
# It is trivially bypassable by adversarial code, see docstring above.
GUARD = r"""
import faulthandler, builtins, os, shutil, subprocess, sys
maximum_memory_bytes = {maximum_memory_bytes}
if maximum_memory_bytes is not None and sys.platform != "darwin":
    # (the resource limit calls seem to fail on macOS, skip them there)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
faulthandler.disable()
builtins.exit = None
builtins.quit = None
builtins.help = None
os.environ["OMP_NUM_THREADS"] = "1"
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    setattr(os, name, None)
for name in ("rmtree", "move", "chown"):
    setattr(shutil, name, None)
subprocess.Popen = None
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None
"""


def execute_code(
    code: str,
    timeout: float = 5.0, # 5 seconds default
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024, # 256MB default
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute as a string
        timeout: Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: Memory limit in bytes (default: 256MB, None to disable)

    Returns:
        ExecutionResult with success status, stdout/stderr, and error information

    Example:
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
    """
    backend = os.getenv("NANOCHAT_EXECUTION_BACKEND", "local")
    if backend == "cwsandbox":
        return _execute_remote(code, timeout, maximum_memory_bytes)
    if backend != "local":
        raise ValueError(f"Unknown NANOCHAT_EXECUTION_BACKEND: {backend}")
    # the guard runs first, then the untrusted code (with fresh globals, as a repr'd literal)
    guard = GUARD.format(maximum_memory_bytes=maximum_memory_bytes)
    program = guard + f"\nexec(compile({code!r}, '<llm>', 'exec'), {{'__name__': '__main__'}})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            process = subprocess.run(
                [sys.executable, "-c", program],
                cwd=tmpdir, # writes land in the tempdir, deleted afterwards
                env={"PATH": "/usr/bin:/bin"}, # scrub the environment
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child process on timeout
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error="Execution timed out (process killed)",
                timeout=True,
            )

    success = process.returncode == 0
    stderr = process.stderr
    # the last line of the traceback identifies the exception, e.g. "TypeError: ..."
    error = None if success else (stderr.strip().splitlines() or ["Execution failed"])[-1]
    memory_exceeded = "MemoryError" in stderr
    result = ExecutionResult(
        success=success,
        stdout=process.stdout,
        stderr=stderr,
        error=error,
        memory_exceeded=memory_exceeded,
    )
    return result


def _execute_remote(code, timeout, maximum_memory_bytes):
    """Fresh container per program. Service/auth errors propagate; never fall back locally."""
    from cwsandbox import AuthStrategy, NetworkOptions, ResourceOptions, Sandbox
    from cwsandbox import SandboxTimeoutError
    from nanochat.observability import trace_span

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    memory = max(512 * 1024 * 1024, maximum_memory_bytes or 0)
    program = GUARD.format(maximum_memory_bytes=maximum_memory_bytes)
    program += f"\nexec(compile({code!r}, '<llm>', 'exec'), {{'__name__': '__main__'}})\n"
    with trace_span("nanochat.execute_code", {"backend": "cwsandbox", "timeout_seconds": timeout}) as output:
        with Sandbox.run(
            auth=AuthStrategy.COREWEAVE_API_KEY if os.getenv("CWSANDBOX_API_KEY") else AuthStrategy.WANDB,
            placement_mode="serverless", # generated Python never runs on CKS GPU containers
            container_image=os.getenv("NANOCHAT_SANDBOX_IMAGE", "python:3.11-slim"),
            max_lifetime_seconds=max(120, math.ceil(timeout) + 60),
            resources=ResourceOptions(requests={"cpu": "1", "memory": str(memory)}),
            network=NetworkOptions(deny_egress=True, deny_ingress=True),
        ) as sandbox:
            output["sandbox_id"] = sandbox.sandbox_id
            sandbox.wait(timeout=60)
            try:
                result = sandbox.exec(["python", "-I", "-c", program],
                                      timeout_seconds=timeout).result()
            except SandboxTimeoutError:
                output.update({"success": False, "timeout": True})
                return ExecutionResult(False, "", "", "Remote execution timed out", timeout=True)
            success = result.returncode == 0
            output.update({"success": success, "returncode": result.returncode})
            error = None if success else (result.stderr.strip().splitlines() or ["Execution failed"])[-1]
            return ExecutionResult(success, result.stdout, result.stderr, error,
                                   memory_exceeded="MemoryError" in result.stderr)
