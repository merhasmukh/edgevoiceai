# Memory & Context Profiles

Full-duplex conversational voice agents often interact with users across multiple turns or multiple days. Remembering user facts (e.g. name, location, preferred language, membership tier) makes the assistant feel intelligent and personal.

`edgevoiceai` separates memory into:
1. **Dialogue History**: Raw chronological turns (`ChatMessage` models).
2. **Entity & Profile State**: Key-value facts extracted during conversations.
3. **Dynamic Prompt Injection**: Automatically formatting profile context into the system prompt.

---

## 1. Memory Store Implementations

`edgevoiceai` provides two standard implementations:

- **`InMemoryMemoryStore`**: Ephemeral, zero-dependency in-memory dictionary. Ideal for testing and single-session interactions.
- **`SQLiteMemoryStore`**: Lightweight file-backed SQL database. Persists conversations and user profiles across application restarts with async queries.

---

## 2. Using `InMemoryMemoryStore`

```python
import asyncio
from edgevoiceai import InMemoryMemoryStore

async def example():
    memory = InMemoryMemoryStore()
    session_id = "user_42"

    # Pre-seed known profile details
    await memory.update_profile(session_id, {
        "user_name": "Rohan Sharma",
        "city": "Mumbai",
        "tier": "Premium Gold",
    })

    # Add dialogue turns
    await memory.add_message(session_id, role="user", content="Hello!")
    await memory.add_message(session_id, role="assistant", content="Hello Rohan! How can I help you today?")

    # Inspect the dynamically generated context prompt
    context_prompt = await memory.build_context_prompt(session_id)
    print(context_prompt)

asyncio.run(example())
```

Output:
```text
[KNOWN USER CONTEXT & PROFILE]
• user_name: Rohan Sharma
• city: Mumbai
• tier: Premium Gold
```

---

## 3. Persistent Memory with `SQLiteMemoryStore`

For production edge devices or servers, use `SQLiteMemoryStore`:

```python
from edgevoiceai import SQLiteMemoryStore

memory = SQLiteMemoryStore(db_path="conversations.db")
```

The database automatically initializes the required `conversations`, `messages`, and `user_profiles` tables.

---

## 4. Automatic Context Injection in `VoiceAgent`

When you pass a memory store to `VoiceAgent`:

```python
agent = VoiceAgent(
    config=config,
    tools=tools,
    memory=memory,
)
```

At the start of every session or turn, `agent.build_effective_system_instruction(session_id)` automatically calls `memory.build_context_prompt(session_id)` and injects the facts directly into Gemini's system instructions.

