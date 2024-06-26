from training_runall import TreeNode, Tree
import pickle

def tree_to_result(tree, true_answer):
    updates = []
    all_nodes = tree.unroll()
    terminal_nodes = [n for n in all_nodes if n.terminal]
    next_up = []
    for node in terminal_nodes:
        # get the true value of the node
        predicted_answer_list = [a[1] for a in tree.answers if a[0] == node]
        if predicted_answer_list: 
            pred = predicted_answer_list[0]
        else:
            pred = 0
        true_value = 1 if pred == true_answer else -1
        estimated_value = node.value
        updates.append((node.state, estimated_value, true_value))
        if node.parent:
            next_up.append((node.parent, true_value))
    # return this interim stuff for now
    return updates, next_up

def save_tree(tree, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(tree, file)

def load_tree(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)