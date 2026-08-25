"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
Adapted from OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

What is covered:
- Each execution runs in its own process (can be killed if it hangs or crashes)
- Execution is limited by a timeout to stop infinite loops
- Memory limits are enforced by default (256MB)
- stdout and stderr are captured and returned
- Code runs in a temporary directory that is deleted afterwards
- Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

What is not covered:
- Not a true security sandbox
- Network access is not blocked (e.g. sockets could be opened)
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- No kernel-level isolation (no seccomp, no containers, no virtualization)

Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, cast

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

    def __repr__(self):
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def capture_io():
    """Capture stdout and stderr, and disable stdin."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None and platform.uname().system != "Darwin":
        # These resource limit calls seem to fail on macOS (Darwin).
        # Also note: `resource` is not available on Windows.
        try:
            import resource
        except ImportError:
            pass
        else:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    faulthandler.disable()

    import builtins as _builtins
    builtins_any = cast(Any, _builtins)
    builtins_any.exit = None
    builtins_any.quit = None
    builtins_any.help = None

    import os as _os
    os_any = cast(Any, _os)

    os_any.environ["OMP_NUM_THREADS"] = "1"

    os_any.kill = None
    os_any.system = None
    os_any.putenv = None
    os_any.remove = None
    os_any.removedirs = None
    os_any.rmdir = None
    os_any.fchdir = None
    os_any.setuid = None
    os_any.fork = None
    os_any.forkpty = None
    os_any.killpg = None
    os_any.rename = None
    os_any.renames = None
    os_any.truncate = None
    os_any.replace = None
    os_any.unlink = None
    os_any.fchmod = None
    os_any.fchown = None
    os_any.chmod = None
    os_any.chown = None
    os_any.chroot = None
    os_any.lchflags = None
    os_any.lchmod = None
    os_any.lchown = None
    os_any.getcwd = None
    os_any.chdir = None

    import shutil as _shutil
    shutil_any = cast(Any, _shutil)

    shutil_any.rmtree = None
    shutil_any.move = None
    shutil_any.chown = None

    import subprocess as _subprocess
    subprocess_any = cast(Any, _subprocess)

    subprocess_any.Popen = None

    import sys as _sys
    modules_any = cast(Any, _sys.modules)

    modules_any["ipdb"] = None
    modules_any["joblib"] = None
    modules_any["resource"] = None
    modules_any["psutil"] = None
    modules_any["tkinter"] = None


def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    """Execute code in a subprocess with safety guards. Results are written to result_dict."""
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # Default to failure
        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            exec_globals = {}
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(code, exec_globals)

            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })

        except TimeoutException:
            result_dict.update({
                "timeout": True,
                "error": "Execution timed out",
            })

        except MemoryError as e:
            result_dict.update({
                "memory_exceeded": True,
                "error": f"Memory limit exceeded: {e}",
            })

        except BaseException as e:
            result_dict.update({
                "error": f"{type(e).__name__}: {e}",
            })

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


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

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        p.join()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",
            timeout=True,
            memory_exceeded=False,
        )

    if not result_dict:
        # The child died BEFORE writing its result dict (RLIMIT_AS MemoryError
        # during import, segfault, reliability-guard kill). That is a crash, not
        # a hang: genuine hangs take the p.is_alive() branch above. Labeling
        # crashes as timeouts made callers' retry/backoff logic keyed on
        # .timeout misfire.
        exitcode = p.exitcode
        detail = (
            f"killed by signal {-exitcode}" if isinstance(exitcode, int) and exitcode < 0
            else f"exitcode={exitcode}"
        )
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error=f"Execution failed (no result returned; {detail})",
            timeout=False,
            memory_exceeded=False,
        )

    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )
