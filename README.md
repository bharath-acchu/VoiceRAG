# VoiceRAG

* It's a Voice RAG project built with LiveKit agent. It uses :

🟢 Silero VAD to detect when I start speaking

🗣️ AssemblyAI to convert my speech to text

📄 ChromaDB for vector search across uploaded documents

🧠 LLM (Gemini/OpenAI) to generate intelligent responses

🗣️ Cartesia TTS to speak the reply out loud

🧬 And finally… Cartesia Voice Cloning so the reply comes in my own voice

❤️ Livekit for orchestration 

## FlowChart
![RAG WORKFLOW](img/img.png)



## Command to run after cloning and installing the requirements

1. If you want to connect and check in liveKitPlayground

```
 uv run voiceRAG.py dev 
```
2. If you want to check in the console(cmd)

```
 uv run voiceRAG.py console 
```

3. If you want to check in the production mode

```
 uv run voiceRAG.py start 
```

Note: 
In the dev and start modes, your agent connects to LiveKit Cloud and joins rooms:

* dev mode: Run your agent in development mode for testing and debugging.

* start mode: Run your agent in production mode.

