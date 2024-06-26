# import from other files
from huggingface_api import generate_response, get_value, tokenizer
from response_processing import process_text_output, process_code, naive_parse
from prompts import prompt_2
import csv
from tqdm import tqdm
import os
from dotenv import load_dotenv
import pickle

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


def generate_responses(text, max_tokens, step_size=100, num_expansions=3, test=False):
    # TODO: Add code block detection/handling for later
    INSIDE_CODE_BLOCK = False
    # INSIDE_CODE_BLOCK = in_code_block(text) 

    # tokenize the text
    tokenized_text = tokenizer(text, return_tensors='pt')
    input_len = len(tokenized_text['input_ids'][0])
    step_size = min(step_size, max_tokens-input_len)

    responses = []

    print(f"\033[92mGenerating {num_expansions} NORMAL responses for input:\n\n{text}\033[0m")
    for i in range(num_expansions):
        # TODO: Make it more efficient by storing old_key_values
        if test:
            print("\033[91mGenerating TEST response actually.\033[0m")
            new_response, old_key_vals, token_length = f"Test response {i+1}.", None
        else:
            new_response, old_key_vals, token_length = generate_response(tokenized_text, INSIDE_CODE_BLOCK, step_size, local=True)
        maybe_answer = process_text_output(new_response)
        answer = None
        if maybe_answer > 0:
            # write in green
            print(f"\033[92mAnswer found: {maybe_answer}\033[0m")
            answer = maybe_answer
        # TODO: Add code block handling
        # if stop_word_cond:
        #     print(f"Stopped: Stop word encountered.")
        # else:
        #     print(f"Stopped: End of generation.")

        terminal = False
        if answer is not None:
            print(f"\033[93mGenerated: {new_response}\033[0m")
            terminal = True

        if token_length == max_tokens:
            print(f"\033[91mStopped this branch: Max tokens reached.\033[0m")
            terminal = True

        gen_response = new_response[len(text):]
        responses.append((gen_response, answer, terminal))
    return responses
    

class TreeNode:
    def __init__(self, state, id=0, parent=None, terminal=False, answer=None):
        self.id = id
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        # currently define value as the inverse of length
        # tensor([[0.0256]], device='cuda:3', dtype=torch.bfloat16,grad_fn=<AddmmBackward0>)
        value_tensor = get_value(tokenizer(state, return_tensors='pt')['input_ids'])
        self.value = value_tensor.item()
        self.terminal = terminal
        self.answer = answer

    def dfs_result(self, true_val):
        return_val = None
        return_list = []
        if self.terminal:
            if self.answer:
                return_val = 1 if self.answer == true_val else -1
            else:
                return_val = 0
            return_list = [(self.state, return_val)]
        elif not self.children:
            return_val = 0
            return_list = [(self.state, return_val)]
        else:
            child_vals = []
            for child in self.children:
                child_list, child_val = child.dfs_result(true_val)
                child_vals.append(child_val)
                return_list.extend(child_list)  
            return_val = sum(child_vals) / len(child_vals)
            return_list.append((self.state, return_val))
        return return_list, return_val

    def __str__(self):
        return self.state
    
    def recursive_print(self, level=0):
        indent = " " * (level * 4)
        state_preview = self.state[:20].replace('\n', '') + '...' + self.state[-20:].replace('\n', '')
        parent_id = self.parent.id if self.parent else None
        print(f"{indent}- State: {state_preview} (Id: {self.id}, Parent: {parent_id}) Value: {self.value:.2f}, DFS value(52): {self.dfs_result(52)[1]}")

        for child in self.children:
            child.recursive_print(level + 1)

    def recursive_visit(self):
        self.visits += 1
        if self.parent:
            self.parent.recursive_visit()

class Tree:
    # a tree structure to store the state of the game
    def __init__(self, text, branching_factor, step_size, max_tokens, max_branching):
        self.root = TreeNode(text)
        self.branching_factor = branching_factor # this defines how much is explored at each step
        self.max_branching = max_branching # this defines how wide each branch can possibly be
        self.c_puct = 1.0
        self.step_size = step_size
        self.max_tokens = max_tokens
        self.next_id = 1
        self.answers = []

    def unroll(self):
        stack = [self.root]
        all_nodes = []
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        return all_nodes
    
    def print_tree(self, level=0):
        # indent = " " * (level * 4)
        # print(f"{indent}- State: {node.state}, Visits: {node.visits}, Value: {node.value:.2f}, Terminal: {node.terminal}")
        # for child in node.children:
        #     self.print_tree(child, level + 1)
        self.root.recursive_print()

    def select(self):
        node = max([n for n in self.unroll() if not n.terminal and len(n.children) < self.max_branching], key=lambda x: x.value + self.c_puct * torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(x.parent.visits if x.parent else 0))/(1+x.visits))))
        # node = max([n for n in self.unroll() if not n.terminal], key=lambda x: x.value)
                   #/x.visits + 2*(2*torch.log(torch.tensor(x.parent.visits))/x.visits)**0.5)
        return node
    
    def expand_node(self, node):
        if os.getenv("TEST") == "True":
            test = True
        else:
            test = False
        expansions = generate_responses(
                        node.state, 
                        self.max_tokens, 
                        step_size=self.step_size, 
                        num_expansions=self.branching_factor,
                        test=test
                    )
        # if all expansions are empty, mark the node as terminal
        if all([expansion[0].strip() == '' for expansion in expansions]):
            node.terminal = True
            return []
        
        for expansion, answer, terminal in expansions:
            print(f"\033[93mCreated child: {expansion}\033[0m")
            new_node = TreeNode(node.state + expansion, id=self.next_id, parent=node, terminal=terminal, answer=answer)
            self.next_id += 1
            if answer:
                self.answers.append((new_node, answer))
            node.children.append(new_node)
        return node.children

    def expand(self):
        expansion_node = None
        try: 
            expansion_node = self.select()
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            return []
        
        if expansion_node:
            expansion_node.recursive_visit()
        # generate branching_factor children
        return self.expand_node(expansion_node)

    def backup(self):
        pass

    def collect_results(self, true_answer):
        updates = []
        all_nodes = self.unroll()
        terminal_nodes = [n for n in all_nodes if n.terminal]
        next_up = []
        for node in terminal_nodes:
            # get the true value of the node
            predicted_answer_list = [a[1] for a in self.answers if a[0] == node]
            if predicted_answer_list: 
                pred = predicted_answer_list[0]
            else:
                pred = 0
            true_value = 1 if pred == true_answer else -1
            estimated_value = node.value
            updates.append((node.state, estimated_value, true_value))
            if node.parent:
                next_up.append((node.parent, true_value))

        while next_up:
            next_iteration = []
            while next_up:
                node, true_value = next_up.pop()
                # check if any of the node's children are also in next_up
                children = [c for c in node.children if c in [n[0] for n in next_up]]
                if children:
                    # if so, add to the list to process in the next iteration
                    next_iteration.append((node, true_value))
                    continue
                # otherwise update the value of the parent
                # get all occurrences of this node in next_up
                all_occurrences = [n for n in next_up if n[0] == node]
                # get the average value of the node
                average_value = sum(n[1] for n in all_occurrences) / len(node.children)
                # update the value of the node
                updates.append((node.state, node.value, average_value))
                if node.parent:
                    next_iteration.append((node.parent, average_value))
            
            # Prepare for the next iteration
            next_up.extend(next_iteration)
        
        return updates

    def mcts(self, n_iter=10):
        answer = None
        for _ in range(n_iter):
            children = self.expand()
        if self.answers:
            answer = sample_best_answer([a[1] for a in self.answers])
        return answer, self.answers

def predict_mcts(problem, branching_factor, max_branching, step_size, max_tokens, n_iter):
    start_text = prompt_2
    start_text = start_text.format(problem)
    tree = Tree(start_text, branching_factor, step_size, max_tokens, max_branching)
    answer, all_answers = tree.mcts(n_iter=n_iter)
    return answer, all_answers, tree

def attempt_training_problem(csv_file, number, mcts, branching_factor, max_branching, n_iter, step_size, max_tokens):
    if mcts:
        print("Running MCTS")
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
                if mcts:
                        predicted_answer, all_answers, tree = predict_mcts(problem, branching_factor, max_branching, step_size, max_tokens, n_iter)
                else:
                        predicted_answer, tree = predict(problem)
                return predicted_answer, tree, answer, all_answers

if __name__ == '__main__':
    # read arguments and parse them
    import argparse
    parser = argparse.ArgumentParser()
    # read the input number
    parser.add_argument('--num_iter', type=int, required=True)
    parser.add_argument('--branching_factor', type=int, default=3)
    parser.add_argument('--max_branching', type=int, default=6)
    parser.add_argument('--mcts', action='store_true')
    parser.add_argument('--num_simulations', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--results_csv_path', type=str, default="results.csv")
    args = parser.parse_args()

    # Path to your CSV file
    csv_file_path = "train.csv"

    # List to hold each problem and answer
    answers = []

    for i in range(args.num_iter):
        try: 
            problem_number = (i % 10) + 1
            print(f"\n\nProblem number: {problem_number}")
            print(f"Iteration: {i+1}")
            answer, final_tree, true_answer, all_answers = attempt_training_problem(
                                                    csv_file_path, 
                                                    problem_number,
                                                    args.mcts, 
                                                    args.branching_factor, 
                                                    args.max_branching, 
                                                    args.num_simulations,
                                                    args.step_size,
                                                    args.max_tokens
                                                ) 
            print(f"\n\nPredicted Answer: {answer}")
            print(f"All answers: {all_answers}")
            print(f"True answer: {true_answer}")
            print(f"\n\n")
            
            final_tree.print_tree()
            count = 1
            for answer in answers:
                print(f"Answer {count}: {answer}")
                count += 1

            # save the tree to a file
            def save_tree(tree, file_name):
                with open(file_name, 'wb') as file:
                    pickle.dump(tree, file)
            
            save_tree(final_tree, f"/opt/dlami/nvme/tree_{problem_number}_{i}_{args.results_csv_path}.pkl")

            results, root_val = final_tree.root.dfs_result(true_answer)

            # append results to the CSV file
            with open(args.results_csv_path, 'a') as file:
                writer = csv.writer(file)
                for result in results:
                    writer.writerow(result)
        except Exception as e:
            print(f"Error on iter {i}: {e}")
            continue

    # # sample value of root node
    # tokenized_text = tokenizer(final_tree.root.state, return_tensors='pt')
    # print(f"Tokenized text input: {tokenized_text}")
    # print(f"Value of input problem: {get_value(tokenized_text['input_ids'])}")
