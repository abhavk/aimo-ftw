# import from other files
from huggingface_api import generate_response
from response_processing import process_text_output, process_code
import csv
from tqdm import tqdm

def sample_best_answer(answers):
    # get the most frequent answer that isn't negative
    answer_counts = {}
    for answer in answers:
        if answer > 0:
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
    
    if len(answer_counts) == 0:
        return -1
    
    best_answer = max(answer_counts, key=answer_counts.get)
    return best_answer

def predict(problem):
    n_repetitions = 15
    answers = []
    for i in tqdm(range(n_repetitions)):
        start_text = """Below is a math problem you are to solve (non-negative numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Then write any code necessary in your solution. Iterate between thoughts and code until you arrive at a solution. Put your final numerical answer within \\boxed{{}}\\.
Note that while the intermediate outputs may be real numbers, the final answer will always be a numerical value."""

        MAX_TOKENS = 2048
        cumulative_text = start_text.format(problem)
        ALREADY_GENERATED = len(cumulative_text)
        # start with approach
        NEXT_GEN = "approach"
        
        while (ALREADY_GENERATED < MAX_TOKENS - 100):
            if NEXT_GEN == "approach":
                cumulative_text += "\n\nApproach:"
            elif NEXT_GEN == "code":
                cumulative_text += "\n\n```python"

            # TODO: This is loose since words and tokens aren't correlated directly
            remaining_words = MAX_TOKENS-len(cumulative_text)
            stop_word_cond = None
            decoded_output = None
            try: 
                decoded_output, stop_word_cond, old_key_values = generate_response(cumulative_text, NEXT_GEN, remaining_words, local=True)
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")

            if stop_word_cond:
                print(f"Stopped: Stop word encountered.")
            else:
                print(f"Stopped: End of generation.")

            generation = decoded_output[len(cumulative_text):]
            print(f"Generated: {generation}")

            if NEXT_GEN == "approach":
                maybe_answer = process_text_output(generation)
                if maybe_answer > 0:
                    print(f"Answer found: {maybe_answer}")
                    answer = maybe_answer
                    
                    break 
                cumulative_text += generation

            else:
                try: 
                    code_output = process_code("```\n"+generation)
                except Exception as e:
                    code_output = str(e)
                cumulative_text += "```output\n" + code_output + "\n```"

        if answer:
            answers.append(maybe_answer)
        else:
            # use the last code output as a potential answer
            answers.append(code_output)

        print(f"Answer: {answer}")
        print(f"Final generation: {cumulative_text}")
        print(f"Current answers: {answers}")
        print(f"\n\n\n")
    
    return sample_best_answer(answers)

def attempt_training_problem(csv_file, number):
    with open(csv_file, 'r') as file:
        # Create a reader object
        csv_reader = csv.DictReader(file)
        # find the problem and answer for the given number
        for i, row in enumerate(csv_reader):
            if i+1 == number:
                problem = row['problem']
                answer = row['answer']
                print(f"Problem: {problem}")
                print(f"True answer: {answer}")
                return predict(problem)


if __name__ == '__main__':
    # read arguments and parse them
    import argparse
    parser = argparse.ArgumentParser()
    # read the input number
    parser.add_argument('--input', type=int, required=True)
    args = parser.parse_args()

    # Path to your CSV file
    csv_file_path = "train.csv"

    # List to hold each problem and answer
    answers = []

    for i in range(15):
        answer = attempt_training_problem(csv_file_path, args.input)
        print(f"Predicted Answer: {answer}")

    count = 1
    for answer in answers:
        print(f"Answer {count}: {answer}")
        count += 1
