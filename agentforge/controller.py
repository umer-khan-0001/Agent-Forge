"""
Agent controller for task planning and execution orchestration.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from agentforge.agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A task to be executed by an agent."""
    id: str
    description: str
    priority: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    assigned_agent: Optional[str] = None


class AgentController:
    """
    Controls and orchestrates multiple agents for complex workflows.
    
    Handles task decomposition, agent assignment, and result aggregation.
    """
    
    def __init__(self):
        """Initialize the agent controller."""
        self.agents: Dict[str, Agent] = {}
        self.tasks: List[Task] = []
        logger.info("Initialized agent controller")
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the controller."""
        self.agents[agent.config.name] = agent
        logger.info(f"Registered agent: {agent.config.name}")
    
    def add_task(self, task_description: str, priority: int = 0) -> Task:
        """Add a task to the queue."""
        import uuid
        task = Task(
            id=str(uuid.uuid4()),
            description=task_description,
            priority=priority
        )
        self.tasks.append(task)
        logger.info(f"Added task: {task.id}")
        return task
    
    def execute_task(
        self,
        task: Task,
        agent_name: Optional[str] = None
    ) -> Any:
        """
        Execute a task with an agent.
        
        Args:
            task: Task to execute
            agent_name: Specific agent to use (or auto-select)
            
        Returns:
            Task execution result
        """
        # Select agent
        if agent_name:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
        else:
            agent = self._select_best_agent(task)
        
        task.status = "in_progress"
        task.assigned_agent = agent.config.name
        
        try:
            result = agent.run(task.description)
            task.status = "completed"
            task.result = result
            return result
        except Exception as e:
            task.status = "failed"
            logger.error(f"Task {task.id} failed: {str(e)}")
            raise
    
    def _select_best_agent(self, task: Task) -> Agent:
        """Select the best agent for a task."""
        # Simple selection - could be more sophisticated
        if not self.agents:
            raise ValueError("No agents registered")
        return list(self.agents.values())[0]
