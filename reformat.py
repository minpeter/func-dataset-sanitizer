import json
from argparse import ArgumentParser
import json


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


def convert_json_to_jsonl(input_file, output_file):
    """Converts JSON array to JSONL."""

    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        if not isinstance(data, list):
            raise ValueError("Input JSON must contain a list of objects.")

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in data:
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write("\n")

        print(
            f"Successfully converted {input_file} to {output_file}"
        )  # Correct indentation here

    except FileNotFoundError:
        print(f"Error: File '{input_file}' or '{output_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage:
input_filename = args.filename
output_filename = "./reformat/" + input_filename.replace(".json", ".jsonl")
convert_json_to_jsonl(input_filename, output_filename)
