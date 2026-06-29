import json
import re
import sys


def convert_signature(sig: str) -> str:
    """
    Convert:
    com.montanha.utils.SenhaUtils.senhaValida:boolean(java.lang.String,java.lang.String)

    Into:
    <com.montanha.utils.SenhaUtils: boolean senhaValida(java.lang.String,java.lang.String)>
    """
    # Match: fully.qualified.Class.method:returnType(parameterTypes)
    pattern = r'^(?P<class_and_method>.+)\:(?P<return_type>[^\(]+)(?P<params>\(.*\))$'
    m = re.match(pattern, sig.strip())
    if not m:
        return sig  # Return unchanged if the signature does not match.

    class_and_method = m.group("class_and_method")
    return_type = m.group("return_type").strip()
    params = m.group("params").strip()

    # Split the class name and method name at the last dot.
    last_dot = class_and_method.rfind(".")
    if last_dot == -1:
        return sig  # Return unchanged if the method-name separator is missing.

    class_name = class_and_method[:last_dot]
    method_name = class_and_method[last_dot + 1:]

    return f"<{class_name}: {return_type} {method_name}{params}>"


def process_json(data):
    """
    Process one JSON object:
    if it contains a call_chain field, convert each method signature in it.
    """
    if isinstance(data, dict):
        if "call_chain" in data and isinstance(data["call_chain"], list):
            data["call_chain"] = [convert_signature(sig) for sig in data["call_chain"]]
        # Recursively process nested fields.
        for k, v in data.items():
            data[k] = process_json(v)
    elif isinstance(data, list):
        data = [process_json(item) for item in data]
    return data


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_call_chain.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = process_json(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Conversion completed. Output file: {output_file}")


if __name__ == "__main__":
    main()
