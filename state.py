from typing import TypedDict, Annotated
from langchain_core.messages import (
    AIMessage, 
    SystemMessage, 
    HumanMessage, 
    ToolMessage,
    BaseMessage
)
from langgraph.graph import add_messages, MessagesState

# class MyState(MessagesState):
#     product_id: Optional[str]
#     mobile_number: Optional[str]
#     email: Optional[str]
#     customer_name: Optional[str]
#     lead_status: Optional[str]

class AdvisorState(TypedDict, total=False):
    messages: Annotated[ list[BaseMessage], add_messages] 
    product_id: str
    mobile_number: str
    email: str
    customer_name: str
    lead_status: str
    summary: str