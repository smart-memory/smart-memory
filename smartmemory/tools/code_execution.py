"""
Code execution tools for Maya assistant.
Provides safe Python code execution and shell command capabilities.
"""
import os
import subprocess
import sys
import tempfile
from typing import Optional


def execute_python(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a safe environment.
    
    Args:
        code (str): Python code to execute
        timeout (int): Execution timeout in seconds (default: 30)
        
    Returns:
        str: Execution result or error message
    """
    try:
        # Basic security checks
        dangerous_imports = ['os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests']
        dangerous_functions = ['exec', 'eval', 'compile', '__import__', 'open', 'file']

        code_lower = code.lower()
        for dangerous in dangerous_imports + dangerous_functions:
            if dangerous in code_lower:
                return f"Error: Potentially dangerous code detected (contains '{dangerous}'). Code execution blocked for security."

        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute the code in a subprocess for isolation
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )

            output = ""
            if result.stdout:
                output += f"Output:\n{result.stdout}\n"
            if result.stderr:
                output += f"Errors:\n{result.stderr}\n"
            if result.returncode != 0:
                output += f"Exit code: {result.returncode}\n"

            return output.strip() if output.strip() else "Code executed successfully (no output)"

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


execute_python.tags = ["code", "python", "execute", "programming"]
execute_python.args_schema = {"code": str, "timeout": int}


def run_shell_command(command: str, timeout: int = 30, working_dir: Optional[str] = None) -> str:
    """
    Execute a shell command safely.
    
    Args:
        command (str): Shell command to execute
        timeout (int): Execution timeout in seconds (default: 30)
        working_dir (str, optional): Working directory for command
        
    Returns:
        str: Command output or error message
    """
    try:
        # Security checks - block dangerous commands
        dangerous_commands = [
            'rm -rf', 'sudo', 'su', 'chmod +x', 'wget', 'curl',
            'dd', 'mkfs', 'fdisk', 'mount', 'umount', 'kill',
            'killall', 'pkill', 'shutdown', 'reboot', 'init'
        ]

        command_lower = command.lower()
        for dangerous in dangerous_commands:
            if dangerous in command_lower:
                return f"Error: Potentially dangerous command detected (contains '{dangerous}'). Execution blocked for security."

        # Additional checks for file system operations
        if any(cmd in command_lower for cmd in ['>', '>>', '|', ';', '&&', '||']):
            return "Error: Command contains potentially dangerous operators. Execution blocked for security."

        # Set working directory
        cwd = working_dir if working_dir else os.getcwd()
        if working_dir and not os.path.exists(working_dir):
            return f"Error: Working directory does not exist: {working_dir}"

        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"Exit code: {result.returncode}\n"

        return output.strip() if output.strip() else "Command executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"


run_shell_command.tags = ["shell", "command", "execute", "system"]
run_shell_command.args_schema = {"command": str, "timeout": int, "working_dir": str}


def install_python_package(package_name: str, version: Optional[str] = None) -> str:
    """
    Install a Python package using pip.
    
    Args:
        package_name (str): Name of the package to install
        version (str, optional): Specific version to install
        
    Returns:
        str: Installation result or error message
    """
    try:
        # Security check - only allow known safe packages
        allowed_packages = [
            'requests', 'beautifulsoup4', 'lxml', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'plotly', 'scipy', 'scikit-learn',
            'pillow', 'opencv-python', 'nltk', 'spacy', 'transformers',
            'torch', 'tensorflow', 'keras', 'jupyter', 'notebook',
            'flask', 'fastapi', 'django', 'sqlalchemy', 'pymongo',
            'redis', 'celery', 'pytest', 'black', 'flake8', 'mypy'
        ]

        if package_name not in allowed_packages:
            return f"Error: Package '{package_name}' is not in the allowed list for security reasons."

        # Build pip command
        pip_command = [sys.executable, '-m', 'pip', 'install']

        if version:
            pip_command.append(f"{package_name}=={version}")
        else:
            pip_command.append(package_name)

        # Execute pip install
        result = subprocess.run(
            pip_command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for package installation
        )

        if result.returncode == 0:
            return f"Successfully installed {package_name}" + (f" version {version}" if version else "")
        else:
            return f"Error installing {package_name}:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return f"Error: Package installation timed out"
    except Exception as e:
        return f"Error installing package {package_name}: {str(e)}"


install_python_package.tags = ["python", "pip", "install", "package"]
install_python_package.args_schema = {"package_name": str, "version": str}


def check_python_syntax(code: str) -> str:
    """
    Check Python code syntax without executing it.
    
    Args:
        code (str): Python code to check
        
    Returns:
        str: Syntax check result
    """
    try:
        compile(code, '<string>', 'exec')
        return "✅ Python syntax is valid"
    except SyntaxError as e:
        return f"❌ Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"
    except Exception as e:
        return f"❌ Error checking syntax: {str(e)}"


check_python_syntax.tags = ["python", "syntax", "check", "validate"]
check_python_syntax.args_schema = {"code": str}


def format_python_code(code: str) -> str:
    """
    Format Python code using basic formatting rules.
    
    Args:
        code (str): Python code to format
        
    Returns:
        str: Formatted code or error message
    """
    try:
        # Try to use black if available
        try:
            import black
            formatted = black.format_str(code, mode=black.FileMode())
            return f"Formatted Python code:\n```python\n{formatted}\n```"
        except ImportError:
            pass

        # Basic formatting fallback
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue

            # Decrease indent for certain keywords
            if stripped.startswith(('except', 'elif', 'else', 'finally')):
                current_indent = max(0, indent_level - 1)
            elif stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                current_indent = indent_level
            else:
                current_indent = indent_level

            formatted_lines.append('    ' * current_indent + stripped)

            # Increase indent after certain keywords
            if stripped.endswith(':') and any(stripped.startswith(kw) for kw in
                                              ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:']):
                indent_level = current_indent + 1
            elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                indent_level = max(0, indent_level - 1)

        formatted_code = '\n'.join(formatted_lines)
        return f"Formatted Python code:\n```python\n{formatted_code}\n```"

    except Exception as e:
        return f"Error formatting code: {str(e)}"


format_python_code.tags = ["python", "format", "style", "beautify"]
format_python_code.args_schema = {"code": str}
