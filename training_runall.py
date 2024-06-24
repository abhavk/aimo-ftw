# import from other files
from huggingface_api import generate_response, tokenizer
from response_processing import process_text_output, process_code, naive_parse
import csv
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

import torch
import gc
import time

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

def predict(problem, max_tokens=2048):
    n_repetitions = 15
    answers = []

    for i in tqdm(range(n_repetitions)):

        print(f"Repetition {i+1}")
        for _ in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
        print(f"Memory allocated: {torch.cuda.memory_allocated()}")

        start_text = """User: Below is a math problem you are to solve (non-negative numerical answer):
\"{}\"
To accomplish this, think carefully and write a short, one-sentence approach for attacking the problem. Then use a sympy-based approach and implement it in python code, surrounded by a ```python{{code}}``` block. Refine your approach and iterate until you are confident of an answer. Put your final numerical answer within \\boxed{{}}\\. Note: While the intermediate outputs may be real numbers, the final answer is always a numerical value."""

        MAX_TOKENS = max_tokens
        cumulative_text = start_text.format(problem)
        ALREADY_GENERATED = len(cumulative_text)
        # start with approach
        NEXT_GEN = "approach"
        answer = None
        while (ALREADY_GENERATED < MAX_TOKENS - 100):
            if NEXT_GEN == "approach":
                cumulative_text = cumulative_text + "\nApproach:"
            # elif NEXT_GEN == "code":
            #     cumulative_text = cumulative_text + "\n```python"

            # TODO: This is loose since words and tokens aren't correlated directly
            remaining_words = MAX_TOKENS-len(cumulative_text)
            stop_word_cond = None
            decoded_output = None
            
            print("\033[92mGenerating response for input:\n" + cumulative_text + "\033[0m")

            try: 
                decoded_output, stop_word_cond, old_key_values = generate_response(cumulative_text, NEXT_GEN, remaining_words, local=True)
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")

            if stop_word_cond:
                print(f"Stopped: Stop word encountered.")
            else:
                print(f"Stopped: End of generation.")

            generation = decoded_output[len(cumulative_text):]
            print(f"\033[93mGenerated: {generation}\033[0m")

            if NEXT_GEN == "approach":
                cumulative_text = cumulative_text + generation
                maybe_answer = process_text_output(generation)
                if maybe_answer > 0:
                    print(f"Answer found: {maybe_answer}")
                    answer = maybe_answer
                    break
                NEXT_GEN = "code"
            else:
                cumulative_text = cumulative_text + generation
                try: 
                    code_output, code_status = process_code("```python\n"+generation, return_shell_output=True)
                    code_output = str(code_output)
                except Exception as e:
                    code_output = str(e)
                cumulative_text = cumulative_text + "\n```output\n" + code_output + "\n```"
                NEXT_GEN = "approach"

            ALREADY_GENERATED = len(cumulative_text)

        if answer:
            answers.append(maybe_answer)
        else:
            # use the last code output as a potential answer
            answers.append(naive_parse(cumulative_text))

        print(f"Answer: {answer}")
        print(f"Final generation: {cumulative_text}")
        print(f"Current answers: {answers}")
        print(f"\n\n\n")
    
    return sample_best_answer([int(a) for a in answers])

def in_code_block(text):
    # basically we want to predict whether we are currently inside a code block
    # find the last occurrence of one of these strings: ```python | ``` | neither
    # if the last occurrence is ```python, we are in code mode
    if "```python" in text:
        # find index of last occurrence
        last_idx = text.rfind("```python")
        if last_idx == -1:
            return False
        # check if it is followed by a ```
        if "```" in text[last_idx:]:
            return False
        else:
            return True
    return False


def generate_responses(text, step_size=100, max_tokens=2048, num_expansions=3, test=False):
    # TODO: Add code block detection/handling for later
    INSIDE_CODE_BLOCK = False
    # INSIDE_CODE_BLOCK = in_code_block(text) 

    # tokenize the text
    tokenized_text = tokenizer(text, return_tensors='pt')
    input_len = len(tokenized_text['input_ids'][0])
    step_size = min(step_size, max_tokens-input_len)

    responses = []

    for i in range(num_expansions):
        # TODO: Make it more efficient by storing old_key_values
        if test:
            print("\033[93mGenerating TEST response for input:\n" + text + "\033[0m")
            new_response, stop_word_cond, stop_word_cond = f"Test response {i+1}.", False, None
        else:
            new_response, stop_word_cond, stop_word_cond = generate_response(tokenized_text, INSIDE_CODE_BLOCK, step_size, local=True)
        maybe_answer = process_text_output(new_response)
        answer = None
        if maybe_answer > 0:
            # write in green
            print("\033[92mAnswer found: {maybe_answer}\033[0m")
            answer = maybe_answer

        # TODO: Add code block handling
        # if stop_word_cond:
        #     print(f"Stopped: Stop word encountered.")
        # else:
        #     print(f"Stopped: End of generation.")
        
        responses.append((new_response, answer))
    return responses
    

class TreeNode:
    def __init__(self, state, parent=None, terminal=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        # currently define value as the inverse of length
        self.value = 1/len(state)
        self.terminal = terminal

class Tree:
    # a tree structure to store the state of the game
    def __init__(self, text, branching_factor, step_size, max_tokens):
        self.root = TreeNode(text)
        self.branching_factor = branching_factor
        self.step_size = step_size
        self.max_tokens = max_tokens
        self.answers = []

    def unroll(self):
        stack = [self.root]
        all_nodes = []
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        return all_nodes
    
    def print_tree(self, node, level=0):
        indent = " " * (level * 4)
        print(f"{indent}- State: {node.state}, Visits: {node.visits}, Value: {node.value:.2f}, Terminal: {node.terminal}")
        for child in node.children:
            self.print_tree(child, level + 1)

    def select(self):
        node = max([n for n in self.unroll() if not n.terminal], key=lambda x: x.value/x.visits + 2*(2*torch.log(torch.tensor(x.parent.visits))/x.visits)**0.5)
        return node
    
    def expand_node(self, node):
        expansions = generate_responses(
                        node.state, 
                        step_size=self.step_size, 
                        max_tokens=self.max_tokens, 
                        num_expansions=self.branching_factor,
                        test=True
                    )
        for expansion, answer in expansions:
            new_node = TreeNode(expansion, parent=node, terminal=answer is not None)
            if answer:
                self.answers.append((new_node, answer))
            node.children.append(new_node)
        return node.children

    def expand(self):
        expansion_node = self.select()
        # generate branching_factor children
        return self.expand_node(expansion_node)

    def backpropagate(self):
        pass

    def mcts(self):
        answer = None
        for _ in range(10):
            children = self.expand()
            for child in children:
                print(f"\033[93mCreated child: {child.state}\033[0m")
            if self.answers:
                answer = sample_best_answer([a[1] for a in self.answers])
                if answer:
                    break

def predict_mcts(problem, branching_factor=3, step_size=100, max_tokens=2048):
    MAX_TOKENS = max_tokens
    start_text = """User: Below is a math problem you are to solve (non-negative numerical answer):
\"{}\"
To accomplish this, think carefully and write a short, one-sentence approach for attacking the problem. Then use a sympy-based approach and implement it in python code, surrounded by a ```python{{code}}``` block. Refine your approach and iterate until you are confident of an answer. Put your final numerical answer within \\boxed{{}}\\. Note: While the intermediate outputs may be real numbers, the final answer is always a numerical value."""
    start_text = start_text.format(problem)
    tree = Tree(start_text, branching_factor, step_size, MAX_TOKENS)
    answer = tree.mcts()
    return answer    
    

def attempt_training_problem(csv_file, number, mcts=False):
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
                if os.getenv("MAX_TOKENS"):
                    print(f"Using MAX_TOKENS: {os.getenv('MAX_TOKENS')}")
                    if mcts:
                        return predict_mcts(problem, int(os.getenv("MAX_TOKENS")))
                    else:
                        return predict(problem, int(os.getenv("MAX_TOKENS")))
                else: 
                    if mcts:
                        return predict_mcts(problem)
                    else:
                        return predict(problem)


if __name__ == '__main__':
    # read arguments and parse them
    import argparse
    parser = argparse.ArgumentParser()
    # read the input number
    parser.add_argument('--input', type=int, required=True)
    parser.add_argument('--mcts', action='store_true')
    args = parser.parse_args()

    # Path to your CSV file
    csv_file_path = "train.csv"

    # List to hold each problem and answer
    answers = []

    for i in range(15):
        answer = attempt_training_problem(csv_file_path, args.input, args.mcts)
        print(f"Predicted Answer: {answer}")

    count = 1
    for answer in answers:
        print(f"Answer {count}: {answer}")
        count += 1
