from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/api/openai/v1", api_key="...")

response = client.completions.create(
    model="claude-3-5-sonnet-20240620",
    prompt="Say this is a test",
    max_tokens=7,
    temperature=0,
)

print(response.choices[0].text)
