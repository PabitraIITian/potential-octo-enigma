import re

from state import AdvisorState
from utils import get_model_from_gcp
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from tools import (
    get_all_products,
    get_product
)
from prompts import SYSTEM_PROMPT

SUMMARY_TRIGGER = 12
KEEP_RECENT = 6

all_tools = [get_product, get_all_products]

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
)
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
PRODUCT_ID_PATTERN = re.compile(r"\bLREAL-[A-Z]+-\d{3}\b", re.IGNORECASE)
NAME_PATTERN = re.compile(
    r"\b(?:my name is|name is|i am called|call me)\s+([^,.!?;\n]+)",
    re.IGNORECASE,
)
NAME_RESPONSE_PATTERN = re.compile(
    r"(?:(?:I am|I'm|This is)\s+)?"
    r"([A-Z][^\W\d_]*(?:[\s'-][A-Z][^\W\d_]*){0,3})[.!]?"
)


def get_model_with_tools(tools):
    llm = get_model_from_gcp()
    return llm.bind_tools(tools=tools)

def assistant(state: AdvisorState):
    llm_with_tools = get_model_with_tools(
        tools = all_tools
    )
    reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] +  state['messages'])
    return {"messages": [reply]}

def safe_cut(messages: list[BaseMessage], keep_recent:int) -> int:
    if keep_recent <= 0:
        return len(messages)

    cut = max(0, len(messages) - keep_recent)
    while cut > 0 and not isinstance(messages[cut], HumanMessage):
        cut -= 1
    return cut


def summarize_converstation(state: AdvisorState):
    messages = state['messages']
    cut = safe_cut(messages, KEEP_RECENT)
    old = messages[:cut]
    instruction = (
        "Summarise this part of a product-advice conversations in under 120 words. "
        "Keep the customers state concerns and which products were discussed. "
        "Do not invent anything"
    )
    previous = state.get('summary')
    if previous:
        instruction = f"Summary so far {previous}\n\n{instruction} Extend it"

    summary = get_model_from_gcp().invoke([SystemMessage(instruction)]).content
    return {
        "summary": summary,
        "messages": [RemoveMessage(id=m.id ) for m in old]

    }
    

def get_tool_node() -> ToolNode:
    """This method returns the tool node

    Returns:
        _type_: _description_
    """
    return ToolNode(tools=all_tools)

def capture_customer_info(state: AdvisorState) -> dict[str, str]:
    messages = state.get("messages", [])
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if last_human_index is None:
        return {}

    customer_message = str(messages[last_human_index].content).strip()
    previous_assistant_message = next(
        (
            str(message.content)
            for message in reversed(messages[:last_human_index])
            if isinstance(message, AIMessage)
        ),
        "",
    )
    updates: dict[str, str] = {}

    email_match = EMAIL_PATTERN.search(customer_message)
    if email_match:
        updates["email"] = email_match.group(0)

    phone_was_requested = bool(
        re.search(r"\b(?:phone|mobile|telephone)\b", previous_assistant_message, re.I)
    )
    phone_was_provided = bool(
        re.search(r"\b(?:phone|mobile|telephone|call|contact)\b", customer_message, re.I)
    )
    if phone_was_requested or phone_was_provided:
        for phone_match in PHONE_PATTERN.finditer(customer_message):
            phone = phone_match.group(0)
            digits = re.sub(r"\D", "", phone)
            if 10 <= len(digits) <= 15:
                updates["mobile_number"] = (
                    f"+{digits}" if phone.lstrip().startswith("+") else digits
                )
                break

    name_match = NAME_PATTERN.search(customer_message)
    if name_match:
        name = re.split(
            r"\s+(?:and|but)\s+(?:my|i|email|phone)\b",
            name_match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if name:
            updates["customer_name"] = name
    elif re.search(r"\bname\b", previous_assistant_message, re.I):
        name_response = NAME_RESPONSE_PATTERN.fullmatch(customer_message)
        if name_response:
            updates["customer_name"] = name_response.group(1)

    product_match = PRODUCT_ID_PATTERN.search(customer_message)
    if product_match:
        updates["product_id"] = product_match.group(0).upper()

    contact_was_requested = bool(
        re.search(r"\b(?:contact|email|phone|mobile)\b", previous_assistant_message, re.I)
    )
    declined = bool(
        re.fullmatch(
            r"(?:no(?:,\s*)?(?:thanks|thank you)?|rather not|not interested|decline)[.! ]*",
            customer_message,
            re.I,
        )
        or re.search(
            r"\b(?:i(?:'d| would) rather not|i (?:do not|don't) want to "
            r"(?:share|provide|give)|not comfortable (?:sharing|providing)|"
            r"prefer not to (?:share|provide)|decline)\b",
            customer_message,
            re.I,
        )
    )
    if contact_was_requested and declined:
        updates["lead_status"] = "declined"
    elif any(field in updates for field in ("email", "mobile_number", "customer_name")):
        updates["lead_status"] = "captured"

    return updates






    
