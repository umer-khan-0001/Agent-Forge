"""
Memory systems for agent context and knowledge retention.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Short-term conversation memory with sliding window.
    
    Maintains recent conversation history for context.
    """
    
    def __init__(self, max_messages: int = 10):
        """Initialize short-term memory."""
        self.max_messages = max_messages
        self.messages: deque[Message] = deque(maxlen=max_messages)
        logger.info(f"Initialized short-term memory with window size {max_messages}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to memory."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            max_tokens: Optional token limit for context
            
        Returns:
            Formatted conversation history
        """
        if not self.messages:
            return ""
        
        context_lines = []
        for msg in self.messages:
            context_lines.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "\n".join(context_lines)
    
    def get_messages(self) -> List[Message]:
        """Get all messages in memory."""
        return list(self.messages)
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        logger.info("Cleared short-term memory")


class LongTermMemory:
    """
    Long-term semantic memory using vector storage.
    
    Stores and retrieves relevant information from past conversations
    and experiences using semantic similarity.
    """
    
    def __init__(self, persist_directory: str = "./chroma_data"):
        """Initialize long-term memory with vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            self.collection = self.client.get_or_create_collection(
                name="agent_memory",
                metadata={"description": "Agent long-term memory"}
            )
            
            logger.info(f"Initialized long-term memory at {persist_directory}")
        except ImportError:
            logger.warning("ChromaDB not available, long-term memory disabled")
            self.client = None
            self.collection = None
    
    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> None:
        """
        Add a memory to long-term storage.
        
        Args:
            content: The memory content
            metadata: Optional metadata
            memory_id: Optional custom ID
        """
        if self.collection is None:
            return
        
        import uuid
        memory_id = memory_id or str(uuid.uuid4())
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[memory_id]
        )
        logger.debug(f"Added memory: {memory_id}")
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant memory contents
        """
        if self.collection is None:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )
        
        if results and results['documents']:
            return results['documents'][0]
        return []
    
    def clear(self) -> None:
        """Clear all memories."""
        if self.collection:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name="agent_memory",
                metadata={"description": "Agent long-term memory"}
            )
            logger.info("Cleared long-term memory")
