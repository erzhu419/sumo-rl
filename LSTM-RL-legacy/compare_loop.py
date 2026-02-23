import ast

def get_while_loop(filename):
    with open(filename, 'r') as f:
        tree = ast.parse(f.read())
        # Find the while not done loop
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # We assume the main while loop is the one we want
                if ast.unparse(node.test) == 'not done':
                     return ast.unparse(node)
    return ""

v_loop = get_while_loop('sac_zero_vanilla.py')
e_loop = get_while_loop('sac_zero_ensemble.py')

if v_loop == e_loop:
    print("The loops are EXACTLY identical in AST structure.")
else:
    print("The loops differ. Writing diff to loop_diff.txt...")
    with open('v_loop.txt', 'w') as f: f.write(v_loop)
    with open('e_loop.txt', 'w') as f: f.write(e_loop)
    import subprocess
    subprocess.run("diff -u v_loop.txt e_loop.txt > loop_diff.txt", shell=True)
