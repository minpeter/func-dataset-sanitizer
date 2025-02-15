from argparse import ArgumentParser
import json


parser = ArgumentParser()
parser.add_argument(
    "-i", "--input", "--inputfile", help="Input file name", dest="input", required=True
)

parser.add_argument(
    "-a",
    "--answer",
    "--answerfile",
    help="Answer file name",
    dest="answer",
    required=True,
)

parser.add_argument(
    "-d",
    "--debug",
    help="Debug mode",
    dest="debug",
    action="store_true",
)

args = parser.parse_args()


def bfcl_ground_truth_to_json(data):

    result = []
    for idx, fc_call in enumerate(data):
        key = list(fc_call.keys())[0]

        args = {}

        for _, value in enumerate(fc_call[key]):
            # If there is no selection for an optional parameter, provide ""
            if fc_call[key][value][0] != "":
                args[value] = fc_call[key][value][0]

        result.append(
            {
                "function": key,
                "arguments": args,
            }
        )
    return result


def parse_function_calling_json(input_data, answer_data):
    if input_data["id"] != answer_data["id"]:
        raise ValueError("ID mismatch")

    parsed_data = {
        "parsed": [
            {
                "from": "human",
                "value": input_data["question"][0][0]["content"],
            },
            {
                "from": "gpt",
                "value": bfcl_ground_truth_to_json(answer_data["ground_truth"]),
            },
        ],
        "tools": input_data["function"],
        "extra": {
            "id": input_data["id"],
        },
    }

    return parsed_data


def process_jsonl_files(input_file_path, answer_file_path, output_file_path):
    """
    Reads a jsonl file, processes each line with parse_function_calling_json,
    and saves the output to a new jsonl file without indentation.
    """

    error_count = 0
    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            with open(answer_file_path, "r", encoding="utf-8") as ansfile:
                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    for input_line, answer_line in zip(infile, ansfile):
                        try:
                            input_data = json.loads(input_line.strip())
                            answer_data = json.loads(answer_line.strip())

                            parsed_data = parse_function_calling_json(
                                input_data, answer_data
                            )

                            json.dump(parsed_data, outfile, ensure_ascii=False)
                            outfile.write("\n")
                        except Exception as e:
                            error_count += 1
                            if args.debug:
                                print(f"Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path} or {output_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")

    total_lines = sum(1 for _ in open(input_file_path, "r", encoding="utf-8"))
    print(
        f"Total lines: {total_lines}, Success: {total_lines - error_count}, Error: {error_count}"
    )


input_file = args.input
answer_file = args.answer

output_file = "./parsed/" + input_file.split(".")[0] + "-parsed.jsonl"

process_jsonl_files(input_file, answer_file, output_file)
