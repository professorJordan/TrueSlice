
import configparser
import csv

############################################################################################################################################

#pull in the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

#The path to the list of questions the LLM tried to answer
QuestionFile = config["CreateResultsFile"]["QuestionFile"]

#short answer file from the dataset
ShortAnswerFile = config["CreateResultsFile"]["ShortAnswerFile"]

#The path to a file containing RAG-influenced answers  
ResponseFile = config["CreateResultsFile"]["ResponseFile"]

#The path to the CSV that will contain short answers and responses for each question
ResultsFile = config["CreateResultsFile"]["ResultsFile"]

############################################################################################################################################

# Open file short answers included with the dataset
with open(ShortAnswerFile, "r", encoding="utf-8") as file:
    Answers = [line.strip() for line in file if line.strip()]

# Open file containing model-generated answers
with open(ResponseFile, "r", encoding="utf-8") as file:
    Responses = [line.strip() for line in file if line.strip()]

# Open file short answers included with the dataset
with open(QuestionFile, "r", encoding="utf-8") as file:
    Questions = [line.strip() for line in file if line.strip()]

# Just a sanity check to ensure both files have the same number of lines
if len(Answers) != len(Responses):
    print(f"Warning: Line count mismatch ({len(Answers)} answers vs {len(Responses)} responses).")



# Open the CSV file which will contain the expected short answer and the RAG/model-generated answer
with open(ResultsFile, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)

    # Write a header row (remove this line if you don't want headers)
    writer.writerow(["Question", "Expected Short Answer", "Model Response", "rating"])
    
    # Loop through both lists
    for question, answer, response in zip(Questions, Answers, Responses):
        # Write the pair as a new row in the CSV
        writer.writerow([question, answer, response])

print(f"Results successfully written to {ResultsFile}")




