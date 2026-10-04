from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    

class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)
    
class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)

    messages: list[ChatMessage] = Field(
        default_factory=list
    )