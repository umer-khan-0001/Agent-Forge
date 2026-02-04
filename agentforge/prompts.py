"""
Prompt templates for agent reasoning and tool calling.
"""

from typing import List, Dict, Any


class PromptTemplate:
    """
    Prompt engineering templates for agent behavior.
    """
    
    def build_react_prompt(
        self,
        tools: str,
        agent_name: str = "Assistant"
    ) -> str:
        """
        Build a ReAct (Reasoning + Acting) system prompt.
        
        Args:
            tools: Description of available tools
            agent_name: Name of the agent
            
        Returns:
            System prompt string
        """
        return f"""You are {agent_name}, an autonomous AI agent that uses the ReAct (Reasoning + Acting) framework to solve tasks.

Your approach:
1. **Thought**: Reason about the current situation and what needs to be done
2. **Action**: Decide which tool to use (if any) and with what input
3. **Observation**: Analyze the result and continue reasoning
4. **Repeat**: Continue this cycle until you can provide a final answer

Available Tools:
{tools}

Guidelines:
- Think step-by-step and be systematic in your approach
- Use tools when you need external information or capabilities
- If you make a mistake, recognize it and correct your approach
- When you have enough information, provide a clear final answer
- Always ground your responses in the observations from tools
- Be concise but thorough in your reasoning

Output Format:
Thought: [Your reasoning about what to do next]
Action: [Tool name and input, or "None" if no action needed]
Observation: [Result from the tool execution]
... (repeat as needed)
Final Answer: [Your complete answer to the task]
"""
    
    def build_tool_selection_prompt(
        self,
        thought: str,
        tools: str
    ) -> str:
        """Build prompt for tool selection."""
        return f"""Given this reasoning step:
"{thought}"

And these available tools:
{tools}

Determine which tool (if any) should be used next. Respond in JSON format:
{{"tool": "tool_name", "input": "input_value"}}

If no tool is needed, respond: {{"tool": null}}
"""
    
    def build_summarization_prompt(
        self,
        conversation: str,
        max_length: int = 200
    ) -> str:
        """Build prompt for conversation summarization."""
        return f"""Summarize the following conversation in {max_length} words or less, preserving key information:

{conversation}

Summary:"""
