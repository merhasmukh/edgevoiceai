# Memory & Session Reference

This section documents conversation session tracking, message representations, in-memory dialogue storage, and persistent SQLite storage.

---

## ChatMessage & ConversationSession

::: edgevoiceai.memory.message
    options:
      show_root_heading: true
      members:
        - ChatMessage
        - ConversationSession

---

## Memory Stores

### BaseMemoryStore

::: edgevoiceai.memory.stores.base
    options:
      show_root_heading: true
      members:
        - BaseMemoryStore

### InMemoryMemoryStore

::: edgevoiceai.memory.stores.memory
    options:
      show_root_heading: true
      members:
        - InMemoryMemoryStore

### SQLiteMemoryStore

::: edgevoiceai.memory.stores.sqlite
    options:
      show_root_heading: true
      members:
        - SQLiteMemoryStore

