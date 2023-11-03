from language_tool_python import LanguageTool


def check_grammar(text):
    tool = LanguageTool('en-US')  # Load English language rules
    matches = tool.check(text)

    # Filter out matches that are simple capital words at the beginning of sentences
    filtered_matches = [str(match) for match in matches if not str(match).startswith("This sentence does not start with an uppercase letter.")]

    return filtered_matches


if __name__ == "__main__":
    # Your recognized text
    recognized_text = "your recognized text goes here."

    # Check grammar
    grammar_matches = check_grammar(recognized_text)

    # Print grammar mistakes (if any)
    if grammar_matches:
        print("Grammar mistakes found:")
        for match in grammar_matches:
            print(match)
    else:
        print("No grammar mistakes found.")
