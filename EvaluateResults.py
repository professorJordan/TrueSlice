import configparser
import csv
import os
import sys

def get_single_key():
    """Reads a single keypress from the terminal without requiring Enter."""
    if os.name == "nt":
        import msvcrt
        key = msvcrt.getch()
        try:
            return key.decode("utf-8").lower()
        except UnicodeDecodeError:
            return ""
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key.lower()

# Pull in the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# The path to the CSV that will contain short answers and responses for each question
results_file = config["CreateResultsFile"]["ResultsFile"]
temp_file = results_file + ".tmp"

if not os.path.exists(results_file):
    print(f"Error: Could not find '{results_file}'. Check your config.ini path.")
    sys.exit(1)

# First pass: count total valid rows and how many are already scored
total_rows = 0
scored_count = 0

with open(results_file, mode="r", newline="", encoding="utf-8") as infile:
    for row in csv.reader(infile):
        if not row:
            continue
        total_rows += 1
        if len(row) >= 4 and row[3].strip().lower() in ("y", "n"):
            scored_count += 1

print(f"Loaded {total_rows} rows ({scored_count} of {total_rows} already scored).")

# Second pass: score remaining rows
with open(results_file, mode="r", newline="", encoding="utf-8") as infile, \
     open(temp_file, mode="w", newline="", encoding="utf-8") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    quit_early = False

    for line_num, row in enumerate(reader, start=1):
        # Skip completely empty rows
        if not row:
            continue

        # Skip if quitting early or if the 4th column already has 'y' or 'n'
        if quit_early or (len(row) >= 4 and row[3].strip().lower() in ("y", "n")):
            writer.writerow(row)
            continue

        # Unpack the first three columns safely
        question = row[0] if len(row) > 0 else ""
        expected_answer = row[1] if len(row) > 1 else ""
        model_response = row[2] if len(row) > 2 else ""

        # Present progress, question, and answers clearly
        print("-" * 60)
        print(f"Progress: {scored_count} of {total_rows} scored | Row {line_num}")
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Actual:   {model_response}")
        print("Match? [y]es / [n]o / [q]uit: ", end="", flush=True)

        # Wait for a valid single keystroke
        while True:
            choice = get_single_key()
            
            if choice in ("y", "n"):
                print(choice)
                updated_row = row[:3] + [choice]
                writer.writerow(updated_row)
                outfile.flush()
                scored_count += 1
                break
            elif choice == "q":
                print(choice)
                print(f"\nSaving progress ({scored_count} of {total_rows} scored) and exiting...")
                writer.writerow(row)
                quit_early = True
                break
            elif choice == "\x03":
                print(f"\nInterrupted! Saving progress ({scored_count} of {total_rows} scored)...")
                writer.writerow(row)
                quit_early = True
                break

# Atomically replace the original file with the updated temporary file
os.replace(temp_file, results_file)
print(f"Done! {scored_count} of {total_rows} scored. All changes saved to {results_file}")

