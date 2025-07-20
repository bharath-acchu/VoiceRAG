import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from llama_index.llms.google_genai import GoogleGenAI
from livekit.agents.job import AutoSubscribe
from livekit.plugins import cartesia, silero, assemblyai,google
from simplerag import create_query_engine
import asyncio

load_dotenv()

# 🔁 Global variable to hold the query engine instance
query_engine = None

@llm.function_tool
async def query_info(query: str) -> str:
    """Get more information about a specific topic"""
    print("Query asked:", query)
    global query_engine
    res = await query_engine.aquery(query)
    print("Query result:", res.response)
    return str(res.response)


async def entrypoint(ctx: JobContext):
    global query_engine

    # Load query_engine once here
    print("🔄 Initializing query engine...")
    query_engine = await create_query_engine()
    print(" Query engine ready!")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = Agent(
        instructions=(
            "You are a voice assistant named Acchu. Your interface "
            "with users will be voice. You should "
            "avoide usage of unpronouncable punctuation." 
            "Make use of tool if required to answer user questions."
        ),
        vad=silero.VAD.load(),
        stt=assemblyai.STT(),
        llm=google.LLM(model="gemini-2.0-flash-exp",temperature=0.8),
        tts=cartesia.TTS(
            model="sonic-turbo",
            voice="6a986928-372f-43ed-8ca1-fa05556167d5",
        ),
        tools=[query_info]
    )

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    await session.say("Hey, how can I help you today?", allow_interruptions=True)


async def async_main():
    global query_engine
    print("⚙️ Preloading query engine in main...")
    query_engine = await create_query_engine()
    print("✅ Query engine ready.")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    print("******************************  STARTED  *******************************************")
    asyncio.run(async_main())
