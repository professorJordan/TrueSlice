
from datasets import Dataset
import configparser

############################################################################################################################################

#pull in the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

ArrowFile = config["RawDataProcessor"]["ArrowFile"]

ShortAnswer = config["RawDataProcessor"]["ShortAnswerFile"]

LongAnswer = config["RawDataProcessor"]["LongAnswerFile"]

Question = config["RawDataProcessor"]["QuestionFile"]

############################################################################################################################################

# Path to downloaded .arrow file
ds = Dataset.from_file(ArrowFile)


def extract_nq_text(example):
  question_text = example["question"]["text"]

  tokens = example["document"]["tokens"]["token"]
  is_html = example["document"]["tokens"]["is_html"]

  # 1. Extract Long Answer (from the first valid annotation)
  long_answer_text = None
  for la in example["annotations"]["long_answer"]:
    start, end = la["start_token"], la["end_token"]
    if start != -1 and end != -1:
      clean_tokens = [
          tokens[i] for i in range(start, end) if not is_html[i]
      ]
      long_answer_text = " ".join(clean_tokens)
      break

  # 2. Extract Short Answer(s)
  short_answers_list = []
  for sa in example["annotations"]["short_answers"]:
    if sa.get("text"):
      short_answers_list.extend(sa["text"])
    elif sa.get("start_token"):
      for start, end in zip(sa["start_token"], sa["end_token"]):
        clean_tokens = [
            tokens[i] for i in range(start, end) if not is_html[i]
        ]
        short_answers_list.append(" ".join(clean_tokens))

  # Check for Yes/No answers if no short span exists
  yes_no = example["annotations"]["yes_no_answer"]
  if not short_answers_list and yes_no:
    if 1 in yes_no:
      short_answers_list.append("YES")
    elif 0 in yes_no:
      short_answers_list.append("NO")

  return {
      "clean_question": question_text,
      "clean_long_answer": long_answer_text,
      "clean_short_answers": list(set(short_answers_list)),
  }


# Process the entire dataset (use ds.select(range(5)) if you only want to test 5 rows)
clean_ds = ds.map(extract_nq_text, remove_columns=ds.column_names)

saved_count = 0

# Open all three files at once so line numbers stay perfectly synchronized
with (
    open(Question, "w", encoding="utf-8") as q_file,
    open(ShortAnswer, "w", encoding="utf-8") as sa_file,
    open(LongAnswer, "w", encoding="utf-8") as la_file,
):
  for row in clean_ds:
    q = row["clean_question"]
    la = row["clean_long_answer"]
    sa_list = row["clean_short_answers"]

    # Check that question, long answer, and at least one short answer all exist
    if q and la and sa_list:
      # Flatten internal newlines/tabs into single spaces so 1 record = 1 line
      q_clean = " ".join(q.split())
      la_clean = " ".join(la.split())
      # Join multiple short answers with " | " (or use sa_list[0] for just the first one)
      sa_clean = " ".join(" | ".join(sa_list).split())

      q_file.write(q_clean + "\n")
      sa_file.write(sa_clean + "\n")
      la_file.write(la_clean + "\n")
      saved_count += 1

print(
    f"Done! Saved {saved_count} aligned records across all three text files."
)
