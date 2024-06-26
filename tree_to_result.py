from training_runall import TreeNode, Tree
import pickle

def tree_to_result(tree):
    tree.print_tree()

def save_tree(tree, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(tree, file)

def load_tree(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)