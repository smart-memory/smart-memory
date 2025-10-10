"""
File operations tools for Maya assistant.
Provides safe file reading, writing, and directory operations.
"""
from pathlib import Path


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path (str): Path to the file to read
        encoding (str): File encoding (default: utf-8)
        
    Returns:
        str: File contents or error message
    """
    try:
        path = Path(file_path).resolve()

        # Basic security check - prevent reading system files
        if str(path).startswith(('/etc/', '/sys/', '/proc/')):
            return f"Error: Access denied to system directory: {file_path}"

        with open(path, 'r', encoding=encoding) as f:
            content = f.read()

        return f"File contents of {file_path}:\n{content}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


read_file.tags = ["file", "read", "io"]
read_file.args_schema = {"file_path": str, "encoding": str}


def write_file(file_path: str, content: str, encoding: str = "utf-8", mode: str = "w") -> str:
    """
    Write content to a file.
    
    Args:
        file_path (str): Path to the file to write
        content (str): Content to write
        encoding (str): File encoding (default: utf-8)
        mode (str): Write mode - 'w' (overwrite) or 'a' (append)
        
    Returns:
        str: Success or error message
    """
    try:
        path = Path(file_path).resolve()

        # Basic security check - prevent writing to system directories
        if str(path).startswith(('/etc/', '/sys/', '/proc/', '/usr/', '/bin/', '/sbin/')):
            return f"Error: Access denied to system directory: {file_path}"

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode, encoding=encoding) as f:
            f.write(content)

        action = "appended to" if mode == "a" else "written to"
        return f"Successfully {action} {file_path} ({len(content)} characters)"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"


write_file.tags = ["file", "write", "io"]
write_file.args_schema = {"file_path": str, "content": str, "encoding": str, "mode": str}


def list_directory(directory_path: str, show_hidden: bool = False) -> str:
    """
    List contents of a directory.
    
    Args:
        directory_path (str): Path to the directory
        show_hidden (bool): Whether to show hidden files (default: False)
        
    Returns:
        str: Directory listing or error message
    """
    try:
        path = Path(directory_path).resolve()

        if not path.exists():
            return f"Error: Directory not found: {directory_path}"

        if not path.is_dir():
            return f"Error: Not a directory: {directory_path}"

        items = []
        for item in sorted(path.iterdir()):
            if not show_hidden and item.name.startswith('.'):
                continue

            item_type = "DIR" if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                try:
                    size = f" ({item.stat().st_size} bytes)"
                except:
                    size = ""

            items.append(f"{item_type}: {item.name}{size}")

        if not items:
            return f"Directory {directory_path} is empty"

        return f"Contents of {directory_path}:\n" + "\n".join(items)
    except PermissionError:
        return f"Error: Permission denied: {directory_path}"
    except Exception as e:
        return f"Error listing directory {directory_path}: {str(e)}"


list_directory.tags = ["file", "directory", "list", "io"]
list_directory.args_schema = {"directory_path": str, "show_hidden": bool}


def file_info(file_path: str) -> str:
    """
    Get information about a file or directory.
    
    Args:
        file_path (str): Path to the file or directory
        
    Returns:
        str: File information or error message
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: Path not found: {file_path}"

        stat = path.stat()

        info = {
            "path": str(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size if path.is_file() else None,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:],
            "exists": True
        }

        if path.is_file():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                info["lines"] = lines
            except:
                info["lines"] = "unknown"

        result = f"File info for {file_path}:\n"
        for key, value in info.items():
            if value is not None:
                result += f"  {key}: {value}\n"

        return result.strip()
    except Exception as e:
        return f"Error getting file info for {file_path}: {str(e)}"


file_info.tags = ["file", "info", "stat", "io"]
file_info.args_schema = {"file_path": str}


def create_directory(directory_path: str, parents: bool = True) -> str:
    """
    Create a directory.
    
    Args:
        directory_path (str): Path to the directory to create
        parents (bool): Create parent directories if needed (default: True)
        
    Returns:
        str: Success or error message
    """
    try:
        path = Path(directory_path).resolve()

        # Basic security check
        if str(path).startswith(('/etc/', '/sys/', '/proc/', '/usr/', '/bin/', '/sbin/')):
            return f"Error: Access denied to system directory: {directory_path}"

        path.mkdir(parents=parents, exist_ok=True)
        return f"Successfully created directory: {directory_path}"
    except PermissionError:
        return f"Error: Permission denied: {directory_path}"
    except Exception as e:
        return f"Error creating directory {directory_path}: {str(e)}"


create_directory.tags = ["file", "directory", "create", "io"]
create_directory.args_schema = {"directory_path": str, "parents": bool}
