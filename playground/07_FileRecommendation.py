import os
from difflib import get_close_matches

def scanMatchFolder(folder_path, user_input):
    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter out directories, keep only files
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    i = 0
    for file in files:
        print(file)
        i =+ 1
        if i >= 10:
            break

    # Check for exact match
    exact_match = [f for f in files if f.lower() == user_input.lower()]

    if exact_match:
        return {"status": "exact_match", "file": exact_match[0]}

    # If no exact match, find closest matches
    closest_matches = get_close_matches(user_input, files, n=3, cutoff=0.4)

    if closest_matches:
        return {"status": "recommendation", "recommendations": closest_matches}
    else:
        return {"status": "no_match", "message": "No matching files found."}

# Example usage
if __name__ == "__main__":
    while True:
        folder_path = input("Enter the folder path: ")
        if not os.path.exists(folder_path):
            continue
        user_input = input("Enter the filename to search for: ")
        if os.path.exists(folder_path):
            break

    result = scanMatchFolder(folder_path, user_input)

    if result["status"] == "exact_match":
        print(f"Exact match found: {result['file']}")
    elif result["status"] == "recommendation":
        print("No exact match found. Did you mean one of these?")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"{i}. {rec}")
    else:
        print(result["message"])
