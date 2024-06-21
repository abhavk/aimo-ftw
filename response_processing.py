import re
import sys
import subprocess

def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)

def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""
    
def pre_process(code):
    # find first occurrence of ``` and only keep code before it
    code = code.split('```')[0]
    # code = "```python\n" + code + "\n```"
    return code

def pre_process_approach_and_code(code):
    # code is wrapped in ```python\n and \n```
    code = code.split('```python')[1]
    code = code.split('\n```')[0]
    return code

def process_code(code, return_shell_output=False):
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    
    code = pre_process_approach_and_code(code)
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    # print in yellow
    print(f"Processed code:")
    if return_shell_output:
        code = code.replace('\n', '\n    ')
            # Add a try...except block
        code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
    
    if not return_shell_output:
        print(code)
    with open('code.py', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        print(shell_output)
        if return_shell_output:
            if return_value=='FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value+='\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value))) % 1000
    except Exception as e:
        print(e,'shell_output')
        code_output = -1
    
    if return_shell_output:
        if code_output==-1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    
    return code_output

def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)

        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]

        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        else:
            result_output = round(float(eval(result_output))) % 1000
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output