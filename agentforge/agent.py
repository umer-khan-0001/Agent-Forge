"""
Core Agent implementation with ReAct pattern and tool calling.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from agentforge.tools import BaseTool, ToolRegistry
from agentforge.memory import ShortTermMemory, LongTermMemory
from agentforge.llm import LLMProvider
from agentforge.prompts import PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    name: str
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_iterations: int = 15
    memory_enabled: bool = True
    self_correction: bool = True
    streaming: bool = False
    verbose: bool = False


@dataclass
class AgentResponse:
    """Response from an agent execution."""
    
    final_answer: str
    thought_process: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None


class Agent:
    """
    Autonomous agent with reasoning, tool calling, and memory.
    
    Implements the ReAct (Reasoning + Acting) pattern for systematic
    problem-solving with self-correction capabilities.
    """
    
    def __init__(
        self,
        name: str,
        llm_model: str = "gpt-4o-mini",
        tools: Optional[List[BaseTool]] = None,
        memory_enabled: bool = True,
        **kwargs
    ):
        """Initialize the agent with tools and configuration."""
        self.config = AgentConfig(
            name=name,
            llm_model=llm_model,
            memory_enabled=memory_enabled,
            **kwargs
        )
        
        # Initialize LLM provider
        self.llm = LLMProvider(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)
        
        # Initialize memory systems
        if self.config.memory_enabled:
            self.short_term_memory = ShortTermMemory()
            self.long_term_memory = LongTermMemory()
        else:
            self.short_term_memory = None
            self.long_term_memory = None
        
        # Prompt templates
        self.prompt_template = PromptTemplate()
        
        logger.info(f"Initialized agent '{self.config.name}' with {len(self.tool_registry)} tools")
    
    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute a task autonomously using ReAct pattern.
        
        Args:
            task: The task description to accomplish
            context: Additional context for the task
            
        Returns:
            AgentResponse with results and execution trace
        """
        logger.info(f"Agent '{self.config.name}' starting task: {task}")
        
        thought_process = []
        actions_taken = []
        tools_used = set()
        
        # Add task to memory
        if self.short_term_memory:
            self.short_term_memory.add_message("user", task)
        
        # Retrieve relevant long-term memories
        relevant_memories = []
        if self.long_term_memory:
            relevant_memories = self.long_term_memory.search(task, top_k=3)
        
        # Build initial prompt with ReAct structure
        system_prompt = self.prompt_template.build_react_prompt(
            tools=self.tool_registry.get_tool_descriptions(),
            agent_name=self.config.name
        )
        
        # Iterative reasoning and acting
        current_state = task
        final_answer = None
        
        for iteration in range(self.config.max_iterations):
            # THOUGHT: Reason about the current state
            thought = self._generate_thought(
                current_state=current_state,
                system_prompt=system_prompt,
                context=context,
                memories=relevant_memories
            )
            thought_process.append(thought)
            
            if self.config.verbose:
                print(f"\n💭 Thought {iteration + 1}: {thought}")
            
            # Check if task is complete
            if self._is_task_complete(thought):
                final_answer = self._extract_final_answer(thought)
                break
            
            # ACTION: Decide which tool to use
            action = self._decide_action(thought, system_prompt)
            
            if action is None:
                # No action needed, use thought as final answer
                final_answer = thought
                break
            
            if self.config.verbose:
                print(f"🔧 Action {iteration + 1}: {action['tool']}({action['input']})")
            
            # Execute the action
            try:
                observation = self._execute_action(action)
                actions_taken.append({
                    "iteration": iteration + 1,
                    "action": action,
                    "observation": observation
                })
                tools_used.add(action['tool'])
                
                if self.config.verbose:
                    print(f"📊 Observation: {observation[:200]}...")
                
                # Update state with observation
                current_state = f"Previous observation: {observation}\n\nContinue solving: {task}"
                
            except Exception as e:
                error_msg = f"Error executing {action['tool']}: {str(e)}"
                logger.error(error_msg)
                
                if self.config.self_correction:
                    # Try to self-correct
                    current_state = f"Error occurred: {error_msg}\n\nTry a different approach for: {task}"
                else:
                    return AgentResponse(
                        final_answer="",
                        thought_process=thought_process,
                        actions_taken=actions_taken,
                        tools_used=list(tools_used),
                        iterations=iteration + 1,
                        success=False,
                        error=error_msg
                    )
        
        # If no final answer was generated, extract from last thought
        if final_answer is None:
            final_answer = self._generate_final_answer(thought_process, actions_taken)
        
        # Store in long-term memory
        if self.long_term_memory:
            self.long_term_memory.add_memory(
                content=f"Task: {task}\nAnswer: {final_answer}",
                metadata={"task": task, "success": True}
            )
        
        # Add to short-term memory
        if self.short_term_memory:
            self.short_term_memory.add_message("assistant", final_answer)
        
        logger.info(f"Agent completed task in {len(actions_taken)} actions")
        
        return AgentResponse(
            final_answer=final_answer,
            thought_process=thought_process,
            actions_taken=actions_taken,
            tools_used=list(tools_used),
            iterations=len(actions_taken),
            success=True
        )
    
    def _generate_thought(
        self,
        current_state: str,
        system_prompt: str,
        context: Optional[Dict[str, Any]],
        memories: List[str]
    ) -> str:
        """Generate a reasoning step."""
        # Build context
        context_str = ""
        if context:
            context_str = f"\nContext: {context}\n"
        
        if memories:
            context_str += f"\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)
        
        # Get conversation history
        history = ""
        if self.short_term_memory:
            history = self.short_term_memory.get_context()
        
        # Generate thought
        prompt = f"{context_str}\n{history}\n\nCurrent state: {current_state}\n\nThought:"
        
        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        return response.strip()
    
    def _is_task_complete(self, thought: str) -> bool:
        """Check if the thought indicates task completion."""
        completion_indicators = [
            "final answer",
            "the answer is",
            "in conclusion",
            "therefore",
            "task completed"
        ]
        thought_lower = thought.lower()
        return any(indicator in thought_lower for indicator in completion_indicators)
    
    def _extract_final_answer(self, thought: str) -> str:
        """Extract the final answer from a completion thought."""
        # Simple extraction - can be improved with better parsing
        if "Final Answer:" in thought:
            return thought.split("Final Answer:")[-1].strip()
        return thought
    
    def _decide_action(self, thought: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Decide which tool to use based on the thought."""
        # Prompt LLM to decide on action
        action_prompt = f"""
Based on this thought: "{thought}"

Available tools:
{self.tool_registry.get_tool_descriptions()}

Which tool should be used? Respond in JSON format:
{{"tool": "tool_name", "input": "tool_input"}}

If no tool is needed, respond with: {{"tool": null}}
"""
        
        response = self.llm.generate(
            system_prompt="You are an action decision system. Respond only with valid JSON.",
            user_prompt=action_prompt
        )
        
        # Parse action (simplified - would use proper JSON parsing)
        if '"tool": null' in response or '"tool":null' in response:
            return None
        
        # Extract tool and input (simplified parsing)
        try:
            import json
            action = json.loads(response)
            return action
        except:
            return None
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute a tool action."""
        tool_name = action['tool']
        tool_input = action['input']
        
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        result = tool.run(tool_input)
        return str(result)
    
    def _generate_final_answer(
        self,
        thought_process: List[str],
        actions_taken: List[Dict[str, Any]]
    ) -> str:
        """Generate a final answer from the execution trace."""
        summary = "Based on the reasoning and actions taken:\n\n"
        summary += "\n".join(f"- {t}" for t in thought_process[-3:])
        
        if actions_taken:
            summary += f"\n\nFinal observations:\n"
            summary += "\n".join(
                f"- {a['observation'][:100]}..." 
                for a in actions_taken[-2:]
            )
        
        return summary
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent."""
        self.tool_registry.register(tool)
        logger.info(f"Registered tool '{tool.name}' with agent '{self.config.name}'")
    
    def clear_memory(self) -> None:
        """Clear all memory systems."""
        if self.short_term_memory:
            self.short_term_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.clear()
        logger.info(f"Cleared memory for agent '{self.config.name}'")
