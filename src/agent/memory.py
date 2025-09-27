from langchain.memory import ConversationBufferMemory

def initialize_memory(context_window: int) -> ConversationBufferMemory:
    """
    Initialize memory for the agent with a specific context window size.
    Inputs:
    - memory_size: The maximum number of turns the memory should hold.

    Outputs:
    - A ConversationBufferMemory object with a context window limit.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",# The key where memory will store the chat history
        k= context_window,
        return_messages=True  # Return messages to include past interaction context
    )
    return memory