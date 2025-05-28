import json
from datasets import load_dataset
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="https://api.friendli.ai/serverless/v1",
    api_key=os.environ.get("FRIENDLI_TOKEN"),
)

ds = load_dataset("./parsed", data_files="*.parquet")
print(ds)

error = []

total = len(ds["train"]["messages"])


def process(idx, messages, tools):
    try:
        completion = client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            # model="gpt-4o-mini",
            messages=messages[:-1],
            # messages=messages,
            tools=json.loads(tools),
            max_tokens=1,
            # max_completion_tokens=1,
        )

        print(completion.choices[0].message)
        return None
    except Exception as e:
        if (
            "Could not finish the message because max_tokens or model output limit was reached."
            in str(e)
        ):
            # print(f"Skipping idx {idx} due to max_tokens error.")
            return None
        else:

            print(f"Index: {idx}")
            print("Messages:", messages)
            print("Tools:", json.loads(tools))

            print(f"\033[91mIdx: {idx}, Error: {e}\033[0m")
            return (idx, str(e))


print("Starting parallel processing...")

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(process, idx, messages, tools)
        for idx, (messages, tools) in enumerate(
            zip(ds["train"]["messages"], ds["train"]["tools"])
        )
    ]
    for f in tqdm(as_completed(futures), total=total, desc="Processing"):
        result = f.result()
        if result:
            error.append(result)

print(f"Total errors: {len(error)}")
print("Errors:", error)
