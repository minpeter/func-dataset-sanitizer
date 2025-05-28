import json
from datasets import load_dataset
import os
from openai import OpenAI

client = OpenAI(
    # base_url="https://api.friendli.ai/serverless/v1",
    # api_key=os.environ.get("FRIENDLI_TOKEN"),
)

ds = load_dataset("./parsed", data_files="*.parquet")
print(ds)

for idx, (messages, tools) in enumerate(
    zip(ds["train"]["messages"], ds["train"]["tools"])
):
    # for debugging
    if idx != 10:  # Limit to first 5 for brevity
        continue
    print(f"Index: {idx}")
    print("Messages:", messages)
    print("Tools:", json.loads(tools))

    completion = client.chat.completions.create(
        # model="meta-llama-3.1-8b-instruct",
        model="gpt-4o-mini",
        # messages=messages,
        messages=[
            {
                "role": "user",
                "content": "You are a helpful assistant.",
            },
        ],
        tools=json.loads(tools),
        max_tokens=1,
    )

    print(completion.choices[0].message)
    print("-" * 40)
