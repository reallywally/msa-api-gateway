from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get wather for a given city"""
    return f"It's always sunny in {city}"


agent = create_agent(
    model="antropic:claude-sonnet-4-5",
    tools=[get_weather],
    system_prompt="you are a helpful assistant",
)


agent.invoke({"messages": [{"role": "user", "content": "what is the wather in sf"}]})
