import os
import re
import openai
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel 
from collections import defaultdict
from tqdm import tqdm
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

BATCH_SIZE = 15

qwen34b_model_path = "your/model/path"
lora_adapter_path = "your/lora/adapter/path" 

output_file = "your/output/file/path" 
benchmark_file = "your/benchmark/file/path"


print(f"Loading tokenizer from: {qwen34b_model_path}")
tokenizer = AutoTokenizer.from_pretrained(qwen34b_model_path, trust_remote_code=True,padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")


print(f"Loading base model from: {qwen34b_model_path}")

model = AutoModelForCausalLM.from_pretrained(
    qwen34b_model_path,
    trust_remote_code=True,
    device_map="auto", 
    torch_dtype=torch.float16
)
print("Base model loaded.")


print(f"Loading LoRA adapters from: {lora_adapter_path}")

model = PeftModel.from_pretrained(model, lora_adapter_path)
print("LoRA adapters loaded onto the base model.")


print("Merging LoRA adapters into the base model for inference...")
model = model.merge_and_unload()
print("LoRA adapters merged and unloaded.")


model.eval()
print("Model set to evaluation mode.")



PROMPT_QWEN34B_TEMPLATE = """You are an expert Python programmer specializing in Operations Research and the Gurobi optimization solver.
Your primary task is to translate the Operations Research problem described below into a complete and runnable Python script using the `gurobipy` library.

**Problem Description:**
{problem_desc}

**Instructions:**
1.**Output code only**. Besides the code, do not include any other text, explanations, or comments. 
2.The code should be complete and runnable without any modifications.
3.Generate the code **only once**, without any modifications or iterations.
4.Keep the code as *simple and clear* as possible, without many comments.

**Final Output (Code Block Only):**
Your *entire* response **MUST** be ONLY a single Python code block, starting with ```python and ending with ```. 
"""

PROMPT_QWENMAX_EVAL_TEMPLATE = """You are an expert evaluator for Operations Research problem solutions. Your task is to rigorously assess whether the 'Provided Answer' below correctly solves the given 'Original OR Problem'. You will also use the 'Standard Answer' provided as a reference for numerical accuracy.

**Core Evaluation Principles:**

1.  **Tolerance for Wording/Format:** The 'Provided Answer' might use different phrasing, formatting (e.g., decimal places, scientific notation), or include supplementary text compared to the 'Standard Answer'. Focus on the informational content and numerical values, not the exact presentation.
2.  **Completeness Check (Based on Original Problem):**
    *   Carefully read the 'Original OR Problem' description to identify *all* specific information items required in the solution (e.g., optimal objective value, number of bins, specific assignments, values of decision variables).
    *   Verify if the 'Provided Answer' contains **all information items requested by the 'Original OR Problem'**. The *type* of information must be present (e.g., if assignments are requested, some form of assignment must be shown).
    *   **Important:** Do *not* mark the answer as incomplete if it lacks extra information present in the 'Standard Answer' but *not* explicitly asked for in the 'Original OR Problem'. Completeness is judged *solely* against the 'Original OR Problem' requirements.
3.  **Numerical Accuracy Check (Based on Standard Answer - Prioritizing Objective):**
    *   **Primary Check: Optimal Objective Value:** Extract the main optimal objective value (e.g., total cost, total profit, **minimum number of bins**) from both the 'Provided Answer' and the 'Standard Answer'.
    *   Normalize these objective values (convert scientific notation, focus on the quantity).
    *   Compare the normalized objective values. They **must be equal** (within a small tolerance, e.g., absolute difference < 1e-5) for the answer to be potentially correct.
    *   **Secondary Check (Conditional):** For problems like Bin Packing (BPP) or others where multiple optimal solutions for decision variables/assignments might exist, if the **optimal objective value matches** the 'Standard Answer', then differences in the specific assignments or non-objective variable values between the 'Provided Answer' and 'Standard Answer' **do not automatically make the answer incorrect**, *unless* the 'Original OR Problem' specifically required a unique representation or format for these secondary details that wasn't met.
    *   For problems where decision variables usually have unique optimal values (like many standard Linear Programs), if the objective matches, *then* also compare any other specific numerical values requested by the 'Original OR Problem' against the 'Standard Answer' (within tolerance).

**Decision Logic:**

*   The 'Provided Answer' is considered **Correct ('Y')** if and only if **both** conditions are met:
    1.  **Completeness:** It includes all information items required by the 'Original OR Problem' (meeting the *type* of information required).
    2.  **Objective Accuracy:** The primary **optimal objective value** reported in the 'Provided Answer' matches the objective value in the 'Standard Answer' (within tolerance).
    *   (Implicitly handled by objective priority): Minor discrepancies in secondary details (like BPP assignments) are tolerated if the objective is correct and completeness is satisfied.
*   If either the Completeness or Objective Accuracy condition is not met, the answer is **Incorrect ('N')**.

--- Input Data ---
**Original OR Problem:**
{problem_desc}

**Provided Answer:**
{result}

**Standard Answer:**
{std_answer}
--- End Input Data ---

**Output Requirement:**

Perform your detailed reasoning and comparison internally based on the principles and input data above. **However, your final output must be strictly a single character:**
*   Output 'Y' if the 'Provided Answer' is judged Correct.
*   Output 'N' if the 'Provided Answer' is judged Incorrect.
**Do not include any other text, explanations, reasoning, spaces, or line breaks in your output.**
"""


def call_qwen34b_batched(prompts: list):
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True, 
        truncation=True,
        return_attention_mask=True
    ).to(model.device) 

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    
    with torch.no_grad(): 
        response_ids_batched = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False, 
        )

    prompt_len_padded = input_ids.shape[1]
    
    decoded_responses = []
    for i in range(response_ids_batched.shape[0]): 
        generated_ids_only = response_ids_batched[i][prompt_len_padded:]
        decoded_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True)
        decoded_responses.append(decoded_text)
        
    return decoded_responses

qwen_config = {
    "base_url": "your_base_url",
    "api_key": os.environ.get("DASHSCOPE_API_KEY", "your_api_key") 
}

def call_qwenmax(config, prompt_content):
    
    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("API key for QwenMax is not set. Please set DASHSCOPE_API_KEY environment variable or update qwen_config.")
    
    client = openai.OpenAI(api_key=api_key, base_url=config["base_url"])
    model_name = "qwen-max"
    messages = [{"role": "user", "content": prompt_content}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling QwenMax API: {e}")
        return "Error_API_Call" 


def extract_problems(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    problems = re.findall(r"--- Problem (\d+) ---\n(.*?)(?=^--- Problem \d+ ---|\Z)", text, re.S | re.MULTILINE)
    processed_problems = []

    for pid, content in problems:
        difficulty_match = re.search(r"\*\*Difficulty:\*\* (.+)", content)
        category_match = re.search(r"\*\*Category:\*\* (.+)", content)
        difficulty = difficulty_match.group(1).strip() if difficulty_match else "Unknown"
        category = category_match.group(1).strip() if category_match else "Unknown"
        
        desc_end_markers = [r"\*\*Modeling Process:\*\*", r"\*\*Answer:\*\*", r"={5,}"]
        desc_pattern = r"\*\*Problem Description:\*\*\s*\n(.*?)(?:" + "|".join(desc_end_markers) + r"|\Z)"
        desc_match_strict = re.search(desc_pattern, content, re.S | re.DOTALL)
        problem_desc = desc_match_strict.group(1).strip() if desc_match_strict else ""
        
        if not problem_desc: 
            desc_match_original = re.search(r"\*\*Problem Description:\*\*\s*\n(.*?)\*\*Modeling Process:\*\*", content, re.S | re.DOTALL)
            problem_desc = desc_match_original.group(1).strip() if desc_match_original else "Error: Could not parse problem description."


        answer_match = re.search(r"Answer:\s*(.*?)\n(?=\n*={5,}|\Z)", content, re.S)
        answer_text = answer_match.group(1).strip() if answer_match else "Error: Could not parse standard answer."

        processed_problems.append({
            "pid": pid,
            "problem_desc": problem_desc,
            "difficulty": difficulty,
            "category": category,
            "std_answer": answer_text})

    return processed_problems

def execute_code(code_with_markdown):
    match = re.search(r"```python\s*(.*?)\s*```", code_with_markdown, re.DOTALL)
    if not match:
        match_no_lang = re.search(r"```\s*(.*?)\s*```", code_with_markdown, re.DOTALL)
        if not match_no_lang:
            
            if "```" not in code_with_markdown: 
                 print(f"DEBUG: No code block markers found. Assuming entire output is Python code. Input was:\n{code_with_markdown[:500]}...")
                 python_code = code_with_markdown.strip()
            else: 
                print(f"DEBUG: No valid Python code block found. Input was:\n{code_with_markdown[:500]}...")
                return "Error: No valid Python code block found in the input."
        else:
            python_code = match_no_lang.group(1).strip()
    else:
        python_code = match.group(1).strip()

    if not python_code:
        
        return "Error: Extracted empty Python code."

    temp_file = "temp_exec.py"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(python_code)

    try:
        
        result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=90)
        output = result.stdout
        if result.stderr:
            
            if "Gurobi Optimizer version" in result.stderr and "Optimal solution found" in output:
                pass 
            elif "Academic license" in result.stderr and "Optimal solution found" in output:
                pass 
            else:
                 output += "\n--- Stderr: ---\n" + result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Execution failed: Timeout after 90 seconds."
    except FileNotFoundError: 
        try:
            result = subprocess.run(["python", temp_file], capture_output=True, text=True, timeout=90)
            output = result.stdout
            if result.stderr:
                if "Gurobi Optimizer version" in result.stderr and "Optimal solution found" in output:
                    pass
                elif "Academic license" in result.stderr and "Optimal solution found" in output:
                    pass
                else:
                    output += "\n--- Stderr: ---\n" + result.stderr
            return output.strip()
        except subprocess.TimeoutExpired:
            return "Execution failed (with python): Timeout after 90 seconds."
        except Exception as e_py:
            return f"Execution failed (with python): {e_py}"
    except Exception as e:
        return f"Execution failed: {e}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    

    all_problem_data = extract_problems(benchmark_file)
    total, correct = 0, 0
    difficulty_stats = defaultdict(lambda: [0, 0])
    category_stats = defaultdict(lambda: [0, 0])
    import gc

    
    if not qwen_config.get("api_key") or qwen_config.get("api_key") == "your_api_key": 
        print("Error: DASHSCOPE_API_KEY is not set or is a placeholder. Please set it as an environment variable.")
        print("Example: export DASHSCOPE_API_KEY='your_api_key'")
        return

    with open(output_file, 'w', encoding='utf-8') as fout:
        for i in tqdm(range(0, len(all_problem_data), BATCH_SIZE), desc="Processing Batches", unit="batch"):
            current_batch_data = all_problem_data[i : i + BATCH_SIZE]
            
            if not current_batch_data: 
                continue

            qwen34b_prompts_batch = [
                PROMPT_QWEN34B_TEMPLATE.format(problem_desc=p_data["problem_desc"])
                for p_data in current_batch_data
            ]
            
            
            if any(not p_data["problem_desc"] or "Error: Could not parse" in p_data["problem_desc"] for p_data in current_batch_data):
                print(f"Warning: Batch starting with Problem {current_batch_data[0]['pid']} contains empty or unparsed problem descriptions. Skipping problematic items or batch if all are bad.")
                
                valid_prompts = []
                valid_data_indices = []
                for idx, p_data in enumerate(current_batch_data):
                    if p_data["problem_desc"] and "Error: Could not parse" not in p_data["problem_desc"]:
                        valid_prompts.append(PROMPT_QWEN34B_TEMPLATE.format(problem_desc=p_data["problem_desc"]))
                        valid_data_indices.append(idx)
                
                if not valid_prompts:
                    print("No valid problems in this batch after filtering. Skipping batch.")
                    continue
                
                qwen34b_prompts_batch = valid_prompts
                
                temp_current_batch_data = [current_batch_data[j] for j in valid_data_indices]
            else:
                temp_current_batch_data = current_batch_data 


            print(f"\n--- Processing Batch starting with Problem {temp_current_batch_data[0]['pid']} ({len(temp_current_batch_data)} problems) ---")

            batched_code_outputs_from_34b = call_qwen34b_batched(qwen34b_prompts_batch)

            if len(batched_code_outputs_from_34b) != len(temp_current_batch_data):
                print(f"Error: Mismatch between number of prompts ({len(temp_current_batch_data)}) and number of generated outputs ({len(batched_code_outputs_from_34b)}). Skipping batch.")
                continue


            for idx_in_batch, problem_data in enumerate(temp_current_batch_data): 
                pid = problem_data["pid"]
                problem_desc = problem_data["problem_desc"]
                difficulty = problem_data["difficulty"]
                category = problem_data["category"]
                std_answer = problem_data["std_answer"]
                
                if "Error: Could not parse" in problem_desc or "Error: Could not parse" in std_answer:
                    print(f"Problem {pid}: Skipping due to parsing error in problem_desc or std_answer.")
                    fout.write(f"--- Problem {pid} ---\n")
                    fout.write(f"Difficulty: {difficulty}\nCategory: {category}\n")
                    fout.write(f"Status: SKIPPED - Parsing Error\n")
                    fout.write(f"Problem Description:\n{problem_desc}\n")
                    fout.write(f"Standard Answer:\n{std_answer}\n\n")
                    fout.flush()
                    total +=1 
                    difficulty_stats[difficulty][1] += 1 
                    category_stats[category][1] += 1   
                    continue

                total += 1 

                code_output_from_34b = batched_code_outputs_from_34b[idx_in_batch]
                execution_result = execute_code(code_output_from_34b)

                current_prompt_qwenmax_eval = PROMPT_QWENMAX_EVAL_TEMPLATE.format(
                    problem_desc=problem_desc,
                    result=execution_result,
                    std_answer=std_answer
                )
                
                correctness = call_qwenmax(qwen_config, current_prompt_qwenmax_eval).strip()
                if correctness == "Error_API_Call":
                    print(f"Problem {pid}: API call to QwenMax failed. Marking as incorrect.")
                    
                
                is_correct = correctness.startswith("Y")
                if is_correct:
                    correct += 1
                    difficulty_stats[difficulty][0] += 1
                    category_stats[category][0] += 1
                difficulty_stats[difficulty][1] += 1
                category_stats[category][1] += 1

                fout.write(f"--- Problem {pid} ---\n")
                fout.write(f"Difficulty: {difficulty}\nCategory: {category}\n")
                fout.write("Problem Description:\n" + problem_desc + "\n\n")
                fout.write("Generated Code (raw from Qwen34B LoRA):\n" + code_output_from_34b + "\n\n")
                fout.write("Execution Output:\n" + execution_result + "\n\n")
                fout.write("Correctness (QwenMax): " + correctness + "\n\n")
                fout.flush()  

                print(f"Problem {pid} Correctness: {correctness}")
                if total > 0 :
                    current_accuracy_value = (correct / total)
                    print(f"Current Accuracy: {correct}/{total} = {current_accuracy_value:.2%}")
                else:
                    print(f"Current Accuracy: {correct}/{total} = 0.00%")
        
    

        fout.write("====== Statistics ======\n")
        if total > 0:
            fout.write(f"Overall Accuracy: {correct}/{total} = {correct/total:.2%}\n\n")
        else:
            fout.write("Overall Accuracy: 0/0 = 0.00%\n\n")


        fout.write("Accuracy by Difficulty:\n")
        for k, (c, t) in sorted(difficulty_stats.items()):
            difficulty_accuracy_value = (c / t) if t > 0 else 0.0
            fout.write(f"{k}: {c}/{t} = {difficulty_accuracy_value:.2%}\n")

        fout.write("\nAccuracy by Category:\n")
        for k, (c, t) in sorted(category_stats.items()):
            category_accuracy_value = (c / t) if t > 0 else 0.0
            fout.write(f"{k}: {c}/{t} = {category_accuracy_value:.2%}\n")

        print("\n====== Statistics ======")
        if total > 0:
            print(f"Overall Accuracy: {correct}/{total} = {correct/total:.2%}")
        else:
            print("Overall Accuracy: 0/0 = 0.00%")
        print("\nAccuracy by Difficulty:")
        for k, (c, t) in sorted(difficulty_stats.items()):
            difficulty_accuracy_value = (c / t) if t > 0 else 0.0
            print(f"{k}: {c}/{t} = {difficulty_accuracy_value:.2%}")
        print("\nAccuracy by Category:")
        for k, (c, t) in sorted(category_stats.items()):
            category_accuracy_value = (c / t) if t > 0 else 0.0
            print(f"{k}: {c}/{t} = {category_accuracy_value:.2%}")
    del batched_code_outputs_from_34b, temp_current_batch_data, qwen34b_prompts_batch
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    main() 