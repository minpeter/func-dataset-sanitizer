from argparse import ArgumentParser
import json, re, ast


parser = ArgumentParser()
parser.add_argument(
    "-f", "--file", "--filename", help="Input file name", dest="filename", required=True
)

parser.add_argument(
    "-d",
    "--debug",
    help="Debug mode",
    dest="debug",
    action="store_true",
)

args = parser.parse_args()


def parse_function_calling_json(data):
    parsed_data = {
        "parsed": [
            {
                "from": "human",
                "value": data["query"],
            },
            {
                "from": "gpt",
                "value": json.loads(data["answers"]),
            },
        ],
        "tools": json.loads(data["tools"]),
        "extra": {
            "id": data["id"],
        },
    }

    return parsed_data


def process_jsonl_files(input_file_path, output_file_path):
    """
    Reads a jsonl file, processes each line with parse_function_calling_json,
    and saves the output to a new jsonl file without indentation.
    """

    error_count = 0
    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line.strip())
                        parsed_data = parse_function_calling_json(data)

                        json.dump(parsed_data, outfile, ensure_ascii=False)
                        outfile.write("\n")
                    except Exception as e:
                        if args.debug:
                            print(f"Error in line {line_num}: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path} or {output_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    total_lines = sum(1 for _ in open(input_file_path, "r", encoding="utf-8"))
    print(
        f"Total lines: {total_lines}, Success: {total_lines - error_count}, Error: {error_count}"
    )


input_file = args.filename
output_file = "./parsed/" + input_file.split(".")[0] + "-parsed.jsonl"

process_jsonl_files(input_file, output_file)
