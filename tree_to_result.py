from training_runall import TreeNode, Tree
import pickle


def save_tree(tree, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(tree, file)

def load_tree(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)