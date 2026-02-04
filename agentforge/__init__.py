"""
AgentForge - Autonomous LLM Agent Framework

A production-ready framework for building intelligent autonomous agents
with tool calling, memory, and multi-agent orchestration.
"""

from agentforge.agent import Agent
from agentforge.controller import AgentController
from agentforge.tools import BaseTool, ToolRegistry

__version__ = "1.0.0"
__author__ = "Umer Khan"

__all__ = [
    "Agent",
    "AgentController",
    "BaseTool",
    "ToolRegistry",
]
