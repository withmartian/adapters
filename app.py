import time
from adapters import AdapterFactory, Prompt
from adapters.types import Conversation, ConversationRole, Turn

# Test for 10000 times and time it

start = time.time()

# for i in range(1000):
adapter = AdapterFactory.get_adapter_by_path(
    "ai21/ai21/jamba-1.5-mini"
)

prompt = Prompt("You are a helpful assistant.")

conv = adapter.convert_to_input(prompt)

res = adapter.execute_sync(conv)

print(res)



# end = time.time()

# print(end - start)

# New: 0.00014829635620117188
# Old: 10.953389167785645
