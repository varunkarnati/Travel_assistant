import os
import json
import uuid
import requests
from datetime import date, datetime
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolInvocation
import operator
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Define fare table
fare_table = {
    ("che", "hyd"): 200,
    ("hyd", "che"): 300,
    ("hyd", "del"): 250,
    # Add more fare entries as needed
}

# Define constants
os.environ["OPENAI_API_KEY"] = "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"
PROJECT = "Demos"
# TAVILY_API_KEY = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
RAPIDAPI_KEY = "d978cd6a3bmsh4351a47d8425e4bp114b3bjsn11185d973e02"
current_time = datetime.now().strftime("%H:%M:%S")
today = date.today()

# Consolidate duplicate fare calculation logic
def calculate_fare(source, destination):
    key = (source, destination)
    return fare_table.get(key, 500)

# Define booking ID map
booking_id_map = {}

# Define tool executor
tools = []

def add_tool(tool):
    tools.append(tool)

# Define tool functions
def get_ride_details(starting_point: str, ending_point: str) -> str:
    fare = calculate_fare(starting_point, ending_point)
    fare_info = {
        "source": starting_point,
        "destination": ending_point,
        "fare": f"{fare}$"
    }
    return json.dumps(fare_info)

def book_ride(starting, ending, time, date) -> str:
    fare = calculate_fare(starting, ending)
    booking_id = str(uuid.uuid4())
    booking_info = {
        "booking id": booking_id,
        "source": starting,
        "destination": ending,
        "time": time,
        "date": date,
        "fare": f"{fare}$"
    }
    booking_id_map[booking_id] = booking_info
    return json.dumps(booking_info)

def get_booking_info(booking_id: str) -> str:
    return json.dumps(booking_id_map.get(booking_id, {}))

def get_flight_info(from_entity: str, to_entity: str, depart_date: str) -> dict:
    def get_sky_id(city):
        url = "https://sky-scanner3.p.rapidapi.com/flights/auto-complete"
        querystring = {"query": city}
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "sky-scanner3.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            data = response.json()
            for item in data['data']:
                if item['presentation']['title'] == city:
                    return item['presentation']['skyId']
        return None
    
    from_entity_id = get_sky_id(from_entity)
    to_entity_id = get_sky_id(to_entity)
    
    url = "https://sky-scanner3.p.rapidapi.com/flights/search-one-way"
    querystring = {
        "fromEntityId": from_entity_id,
        "toEntityId": to_entity_id,
        "departDate": depart_date
    }
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "sky-scanner3.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()


# Define classes for tool input schemas
class Enquire(BaseModel):
    """When user requires an enquiry on a ride """
    starting_point: str = Field(..., description="City from where user starts from, NOT  NULL.")
    ending_point: str = Field(..., description="City to where user wants to travel to, NOT NULL.")

class Book(BaseModel):
    """When user's intent is to book a ride from one place to another only at a specified date and time"""
    starting: str = Field(..., description="The city from where the ride starts from, NOT NULL.")
    ending: str = Field(..., description="The city from where the ride ends, NOT NULL.")
    time: str = Field(..., description=f"Time when the ride has to be booked in 24hr format, should be after {current_time}")
    date: str = Field(..., description=f"Date when the booking needs to be done, in DDMMYYYY format, should be after {today}")

class Flight(BaseModel):
    from_entity: str = Field(description="From city")
    to_entity: str = Field(description="To city")
    depart_date: str = Field(description=f"Date of departure in the YYYY-MM-DD format, should be after {today}")

class RideInfo(BaseModel):
    booking_id: str = Field(description="UUID assigned to the ride")



# Define tool setup
add_tool(StructuredTool.from_function(
    func=get_ride_details,
    name="Get_ride_details",
    description="Get the details of a ride from starting point to ending point",
    args_schema=Enquire,
    return_direct=False,
))

add_tool(StructuredTool.from_function(
    func=book_ride,
    name="Book_ride",
    description="Book a ride between two stations at a specified time and date",
    args_schema=Book,
    return_direct=False,
))

add_tool(StructuredTool.from_function(
    func=get_booking_info,
    name="Get_booking_info",
    description="Get the details of the booking with an ID",
    args_schema=RideInfo,
    return_direct=False,
))

add_tool(StructuredTool.from_function(
    func=get_flight_info,
    name="Get_flight",
    description="Get the flight details from a city to a city on a specified date",
    args_schema=Flight,
    return_direct=False,
))

# Define the OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

tool_executor = ToolExecutor(tools)


# Define class for response schema
class Response(BaseModel):
    """Final answer to the user"""

    brief_response: str = Field(description="Briefing of the response for the user's request")
    explanation: str = Field(description="Detailed description of the response to the user's request")

# Define the model functions
functions = [convert_to_openai_function(t) for t in tools]
functions.append(convert_to_openai_function(Response))

model = {"messages": RunnablePassthrough()} | ChatPromptTemplate.from_messages([("system", "You are a helpful assistant and never hallucinate"), MessagesPlaceholder(variable_name="messages", optional=True)]) | llm.bind_functions(functions)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    elif last_message.additional_kwargs["function_call"]["name"] == "Response":
        return "end_1"
    # Otherwise if there is, we continue
    else:
        return "continue"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define tool invocation function
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

# Define graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", call_tool)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
graph.add_edge("action", "agent")
app = graph.compile()

# Initialize conversation
conversation = []

# Run the conversation loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    conversation.append(HumanMessage(content=user_input))
    inputs = {"messages": conversation}
    for output in app.with_config({"recursion_limit": 50, "run_name": "LLM with Functions"}).stream(inputs):
        for key, value in output.items():
            if key == "agent":
                conversation.append(AIMessage(content=value['messages'][0].content))
                print(value['messages'][0].content)
