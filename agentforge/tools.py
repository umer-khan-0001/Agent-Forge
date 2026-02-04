"""
Base tool classes and tool registry for agent tool calling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Base class for all agent tools.
    
    Tools are functions that agents can call to interact with
    external systems, perform computations, or retrieve information.
    """
    
    name: str = ""
    description: str = ""
    
    def __init__(self):
        """Initialize the tool."""
        if not self.name:
            self.name = self.__class__.__name__.replace("Tool", "").lower()
        if not self.description:
            self.description = self.__doc__ or "No description provided"
    
    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """
        Execute the tool logic.
        
        Must be implemented by subclasses.
        """
        pass
    
    def run(self, input_str: str) -> Any:
        """
        Execute the tool with the given input.
        
        Args:
            input_str: The input to the tool
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"Executing tool '{self.name}' with input: {input_str[:100]}")
            result = self._run(input_str)
            logger.info(f"Tool '{self.name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {str(e)}")
            raise
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's input/output schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input to the tool"
                    }
                },
                "required": ["input"]
            }
        }


class CalculatorTool(BaseTool):
    """Perform mathematical calculations and evaluate expressions."""
    
    name = "calculator"
    description = "Useful for mathematical calculations. Input should be a valid mathematical expression."
    
    def _run(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            import sympy
            result = sympy.sympify(expression)
            evaluated = result.evalf()
            return f"The result of {expression} is {evaluated}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


class WebSearchTool(BaseTool):
    """Search the web for current information."""
    
    name = "web_search"
    description = "Search the web for current information. Input should be a search query."
    
    def _run(self, query: str) -> str:
        """Perform a web search."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(f"Title: {r['title']}\n{r['body']}\nURL: {r['href']}")
            
            if results:
                return "\n\n---\n\n".join(results)
            return "No results found"
        except Exception as e:
            return f"Search failed: {str(e)}"


class WikipediaTool(BaseTool):
    """Query Wikipedia for encyclopedic information."""
    
    name = "wikipedia"
    description = "Query Wikipedia for information. Input should be a topic or question."
    
    def _run(self, query: str) -> str:
        """Query Wikipedia."""
        try:
            import wikipedia
            wikipedia.set_lang("en")
            
            # Search for the page
            search_results = wikipedia.search(query, results=1)
            if not search_results:
                return "No Wikipedia article found"
            
            # Get summary
            page = wikipedia.page(search_results[0], auto_suggest=False)
            summary = wikipedia.summary(search_results[0], sentences=3)
            
            return f"Wikipedia: {page.title}\n\n{summary}\n\nURL: {page.url}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple results found. Please be more specific. Options: {', '.join(e.options[:5])}"
        except Exception as e:
            return f"Wikipedia query failed: {str(e)}"


class FileReadTool(BaseTool):
    """Read contents of a file."""
    
    name = "file_read"
    description = "Read the contents of a file. Input should be a file path."
    
    def _run(self, file_path: str) -> str:
        """Read a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File contents:\n\n{content}"
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class FileWriteTool(BaseTool):
    """Write content to a file."""
    
    name = "file_write"
    description = "Write content to a file. Input format: 'filepath|content'"
    
    def _run(self, input_str: str) -> str:
        """Write to a file."""
        try:
            parts = input_str.split('|', 1)
            if len(parts) != 2:
                return "Error: Input must be in format 'filepath|content'"
            
            file_path, content = parts
            with open(file_path.strip(), 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class PythonREPLTool(BaseTool):
    """Execute Python code safely."""
    
    name = "python_repl"
    description = "Execute Python code. Input should be valid Python code."
    
    def _run(self, code: str) -> str:
        """Execute Python code."""
        try:
            # Create a restricted namespace
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "sum": sum,
                    "max": max,
                    "min": min,
                }
            }
            
            # Capture stdout
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                exec(code, namespace)
                output = buffer.getvalue()
            finally:
                sys.stdout = old_stdout
            
            if output:
                return f"Output:\n{output}"
            return "Code executed successfully (no output)"
        except Exception as e:
            return f"Error executing code: {str(e)}"


class ToolRegistry:
    """
    Registry for managing and accessing tools.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools."""
        if not self.tools:
            return "No tools available"
        
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def list_tools(self) -> List[str]:
        """Get list of all tool names."""
        return list(self.tools.keys())
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self.tools)
