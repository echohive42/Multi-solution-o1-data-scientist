import json
import os
import signal
import subprocess
import sys
import time
import asyncio
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from openai import AsyncOpenAI
from termcolor import colored

# Constants
MODEL = "o1"
ERROR_CORRECTION_MODEL = "o1-mini"
HOW_MANY_SOLUTIONS = 3
INPUT_FILE = "additional_info.txt"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSIONS_DIR = "submissions"
METRICS_DIR = "metrics"
SOLUTIONS_DIR = "solutions"

# Add folder structure constants
OUTPUT_FILES = [f"{SOLUTIONS_DIR}/solution_{i+1}.py" for i in range(HOW_MANY_SOLUTIONS)]
METRICS_FILES = [f"{METRICS_DIR}/metrics_solution_{i+1}.json" for i in range(HOW_MANY_SOLUTIONS)]

MAX_RUNTIME_MINUTES = 5
MAX_ERROR_CORRECTION_ATTEMPTS = 5
IS_WINDOWS = sys.platform.startswith('win')
SAMPLE_ROWS = 30
REASONING_EFFORT = "high"

# Add new constants at the top with other constants
MAX_ITERATIONS = 3 # Number of improvement iterations
ITERATIONS_DIR = "iterations"  # To store iteration history

# Add new constant for token usage file
TOKEN_USAGE_FILE = "token_usage.json"

def get_process_group_kwargs():
    """Get the appropriate kwargs for process group handling based on OS"""
    if IS_WINDOWS:
        return {
            'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP
        }
    return {
        'preexec_fn': os.setsid
    }

def kill_process_tree(process):
    """Kill process tree appropriately for the OS"""
    try:
        if IS_WINDOWS:
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception as e:
        print(colored(f"Error killing process: {str(e)}", "red"))

def handle_sigint(signum, frame):
    """Handle Ctrl+C by terminating all child processes"""
    print(colored("\nCtrl+C detected. Terminating all processes...", "yellow"))
    if IS_WINDOWS:
        # On Windows, we need to handle this differently
        sys.exit(1)
    else:
        os.killpg(os.getpgid(0), signal.SIGTERM)
        sys.exit(1)

def get_data_sample():
    """Get a sample of the training data with labels"""
    try:
        print(colored("Reading training data sample...", "cyan"))
        df = pd.read_csv(TRAIN_FILE)
        sample = df.sample(n=SAMPLE_ROWS, random_state=42)
        
        # Convert sample to formatted string
        sample_str = "Sample Training Data (first few rows with labels):\n"
        sample_str += sample.to_string()
        sample_str += "\n\nColumn Names and their meanings:\n"
        for col in df.columns:
            sample_str += f"- {col}\n"
            
        # Add some basic statistics
        sample_str += "\nBasic Dataset Statistics:\n"
        sample_str += f"Total number of records: {len(df)}\n"
        sample_str += f"Number of features: {len(df.columns) - 1}\n"  # Excluding target
        sample_str += f"Target variable: 'Transported' (True/False)\n"
        
        return sample_str
    except Exception as e:
        print(colored(f"Error reading training data: {str(e)}", "red"))
        return ""

def read_challenge_info():
    """Read challenge info and combine with data sample"""
    try:
        print(colored("Reading challenge information...", "cyan"))
        with open(INPUT_FILE, encoding="utf-8") as f:
            challenge_info = f.read()
            
        # Get data sample
        data_sample = get_data_sample()
        
        # Combine challenge info with data sample
        complete_info = f"{challenge_info}\n\n{data_sample}"
        return complete_info
    except Exception as e:
        print(colored(f"Error reading file: {str(e)}", "red"))
        return None

async def fix_solution(client, solution_file, error_message):
    """Attempt to fix a solution using O1-mini"""
    try:
        print(colored(f"\nAttempting to fix {solution_file}...", "cyan"))
        
        with open(solution_file, 'r', encoding='utf-8') as f:
            current_code = f.read()
        
        # Get the complete challenge info including data sample
        challenge_info = read_challenge_info()
        
        # Take only the last 500 characters of the error message
        truncated_error = error_message[-500:] if len(error_message) > 500 else error_message
        if len(error_message) > 500:
            truncated_error = "..." + truncated_error
        
        prompt = f"""
        Fix the following Python code that generated this error.
        
        Challenge Context:
        {challenge_info}
        
        ERROR:
        {truncated_error}
        
        CODE:
        {current_code}
        
        Return only the fixed code, ensuring it:
        is returned in between ```python and ```
        1. Maintains all existing functionality
        2. Fixes the error
        3. Follows best practices
        4. Includes proper error handling
        5. Works with the provided dataset structure
        """
        
        response = await client.chat.completions.create(
            model=ERROR_CORRECTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"}
        )
        
        fixed_code = response.choices[0].message.content
        
        # Extract code from between ```python and ``` tags
        try:
            if "```python" in fixed_code and "```" in fixed_code:
                # Split by ```python and take the latter part
                code_parts = fixed_code.split("```python", 1)
                if len(code_parts) > 1:
                    # Split by ``` and take the first part
                    fixed_code = code_parts[1].split("```", 1)[0].strip()
                    print(colored("Successfully extracted code from markdown", "green"))
            else:
                print(colored("No code blocks found, using response as-is", "yellow"))
        except Exception as e:
            print(colored(f"Error extracting code from markdown: {str(e)}", "yellow"))
        
        # Save the fixed solution
        with open(solution_file, 'w', encoding='utf-8') as f:
            f.write(fixed_code)
            
        print(colored(f"Fixed code saved to {solution_file}", "green"))
        return True
        
    except Exception as e:
        print(colored(f"Error fixing solution: {str(e)}", "red"))
        return False

async def run_solution_with_fixes(client, solution_file, attempt=1):
    """Run a single solution with error correction"""
    try:
        print(colored(f"Starting {solution_file} (attempt {attempt}/{MAX_ERROR_CORRECTION_ATTEMPTS})...", "cyan"))
        
        # Get OS-appropriate process creation arguments
        process_kwargs = get_process_group_kwargs()
        
        process = subprocess.Popen(
            ['python', solution_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **process_kwargs
        )
        
        try:
            stdout, stderr = process.communicate(timeout=MAX_RUNTIME_MINUTES * 60)
            
            if process.returncode == 0:
                print(colored(f"{solution_file} completed successfully", "green"))
                return True
            else:
                print(colored(f"{solution_file} failed with error:", "yellow"))
                print(colored(stderr, "yellow"))
                
                if attempt < MAX_ERROR_CORRECTION_ATTEMPTS:
                    # Attempt to fix the error
                    fixed = await fix_solution(client, solution_file, stderr)
                    if fixed:
                        # Recursively try running the fixed solution with incremented attempt counter
                        return await run_solution_with_fixes(client, solution_file, attempt + 1)
                    else:
                        print(colored(f"Failed to fix {solution_file} after attempt {attempt}", "red"))
                        return False
                else:
                    print(colored(f"{solution_file} failed after {MAX_ERROR_CORRECTION_ATTEMPTS} attempts", "red"))
                    return False
                
        except subprocess.TimeoutExpired:
            print(colored(f"{solution_file} timed out after {MAX_RUNTIME_MINUTES} minutes", "yellow"))
            kill_process_tree(process)
            if attempt < MAX_ERROR_CORRECTION_ATTEMPTS:
                # Try to fix timeout issues
                timeout_error = f"Process timed out after {MAX_RUNTIME_MINUTES} minutes"
                fixed = await fix_solution(client, solution_file, timeout_error)
                if fixed:
                    return await run_solution_with_fixes(client, solution_file, attempt + 1)
            return False
            
    except Exception as e:
        print(colored(f"Error running {solution_file}: {str(e)}", "red"))
        if attempt < MAX_ERROR_CORRECTION_ATTEMPTS:
            fixed = await fix_solution(client, solution_file, str(e))
            if fixed:
                return await run_solution_with_fixes(client, solution_file, attempt + 1)
        return False

async def run_solutions_parallel():
    """Run all solutions in parallel with error correction"""
    try:
        print(colored(f"\nRunning solutions in parallel (timeout: {MAX_RUNTIME_MINUTES} minutes)...", "cyan"))
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, handle_sigint)
        
        client = AsyncOpenAI()
        
        # Run all solutions concurrently
        tasks = [run_solution_with_fixes(client, solution_file) for solution_file in OUTPUT_FILES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful completions
        completed = [
            file for file, success in zip(OUTPUT_FILES, results) 
            if isinstance(success, bool) and success
        ]
        
        print(colored(f"\nCompleted {len(completed)} out of {len(OUTPUT_FILES)} solutions", "cyan"))
        
    except Exception as e:
        print(colored(f"Error in parallel execution: {str(e)}", "red"))

def setup_directories():
    """Create necessary directories if they don't exist"""
    try:
        print(colored("Setting up directory structure...", "cyan"))
        # Create base iteration directories
        for i in range(1, MAX_ITERATIONS + 1):
            os.makedirs(f"{SOLUTIONS_DIR}/iteration_{i}", exist_ok=True)
            os.makedirs(f"{METRICS_DIR}/iteration_{i}", exist_ok=True)
            os.makedirs(f"{SUBMISSIONS_DIR}/iteration_{i}", exist_ok=True)
        print(colored("Directory structure created successfully", "green"))
    except Exception as e:
        print(colored(f"Error creating directories: {str(e)}", "red"))
        raise

def save_solutions(solutions):
    """Save solutions to files with proper formatting"""
    try:
        print(colored("Saving solutions to files...", "cyan"))
        for file_name, solution in zip(OUTPUT_FILES, solutions):
            # Convert string literal newlines to actual newlines and fix indentation
            formatted_solution = (
                solution
                .replace('\\n', '\n')
                .replace('\\"', '"')
                .strip()
            )
            
            try:
                compile(formatted_solution, file_name, 'exec')
            except Exception as e:
                print(colored(f"Warning: Code validation failed for {file_name}: {str(e)}", "yellow"))
            
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(formatted_solution)
            print(colored(f"Saved solution to {file_name}", "green"))
    except Exception as e:
        print(colored(f"Error saving solutions: {str(e)}", "red"))

def save_token_usage(response, iteration):
    """Save token usage information with iteration details"""
    try:
        # Load existing token usage data if file exists
        token_data = {}
        if os.path.exists(TOKEN_USAGE_FILE):
            with open(TOKEN_USAGE_FILE, 'r', encoding='utf-8') as f:
                token_data = json.load(f)
        
        # Create iteration entry if it doesn't exist
        if 'iterations' not in token_data:
            token_data['iterations'] = {}
        
        # Get usage statistics
        usage = response.usage
        iteration_data = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add reasoning tokens if available
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            iteration_data.update({
                'reasoning_tokens': details.reasoning_tokens,
                'accepted_prediction_tokens': details.accepted_prediction_tokens,
                'rejected_prediction_tokens': details.rejected_prediction_tokens
            })
        
        # Save to iteration data
        token_data['iterations'][f'iteration_{iteration}'] = iteration_data
        
        # Calculate and update totals
        totals = {
            'total_prompt_tokens': sum(iter_data['prompt_tokens'] for iter_data in token_data['iterations'].values()),
            'total_completion_tokens': sum(iter_data['completion_tokens'] for iter_data in token_data['iterations'].values()),
            'total_tokens': sum(iter_data['total_tokens'] for iter_data in token_data['iterations'].values()),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        token_data['totals'] = totals
        
        # Save to file
        with open(TOKEN_USAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(token_data, f, indent=2)
            
        print(colored("\nToken Usage Statistics:", "cyan"))
        print(colored(f"Prompt tokens: {usage.prompt_tokens}", "yellow"))
        print(colored(f"Completion tokens: {usage.completion_tokens}", "yellow"))
        print(colored(f"Total tokens: {usage.total_tokens}", "yellow"))
        print(colored(f"Token usage saved to {TOKEN_USAGE_FILE}", "green"))
        
    except Exception as e:
        print(colored(f"Error saving token usage: {str(e)}", "red"))

def get_iteration_paths(iteration):
    """Get paths for the current iteration"""
    return {
        'solutions': [f"{SOLUTIONS_DIR}/iteration_{iteration}/solution_{i+1}.py" 
                     for i in range(HOW_MANY_SOLUTIONS)],
        'metrics': [f"{METRICS_DIR}/iteration_{iteration}/metrics_solution_{i+1}.json" 
                   for i in range(HOW_MANY_SOLUTIONS)],
        'submissions': [f"{SUBMISSIONS_DIR}/iteration_{iteration}/submission_{i+1}.csv" 
                      for i in range(HOW_MANY_SOLUTIONS)]
    }

def collect_previous_results(iteration):
    """Collect results from previous iteration for improvement"""
    if iteration <= 1:
        return None
    
    try:
        prev_iteration = iteration - 1
        results = []
        
        for i in range(HOW_MANY_SOLUTIONS):
            solution_path = f"{SOLUTIONS_DIR}/iteration_{prev_iteration}/solution_{i+1}.py"
            metrics_path = f"{METRICS_DIR}/iteration_{prev_iteration}/metrics_solution_{i+1}.json"
            submission_path = f"{SUBMISSIONS_DIR}/iteration_{prev_iteration}/submission_{i+1}.csv"
            
            solution_result = {
                'solution_number': i + 1,
                'code': '',
                'metrics': {},
                'submission_sample': None
            }
            
            # Read solution code
            try:
                with open(solution_path, 'r', encoding='utf-8') as f:
                    solution_result['code'] = f.read()
            except Exception as e:
                print(colored(f"Error reading solution {i+1}: {str(e)}", "yellow"))
            
            # Read metrics
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    solution_result['metrics'] = json.load(f)
            except Exception as e:
                print(colored(f"Error reading metrics for solution {i+1}: {str(e)}", "yellow"))
            
            # Read submission sample (first few rows)
            try:
                submission_df = pd.read_csv(submission_path)
                solution_result['submission_sample'] = submission_df.head().to_string()
            except Exception as e:
                print(colored(f"Error reading submission for solution {i+1}: {str(e)}", "yellow"))
            
            results.append(solution_result)
        
        return results
    except Exception as e:
        print(colored(f"Error collecting previous results: {str(e)}", "red"))
        return None

async def async_main():
    try:
        # Set up directory structure first
        setup_directories()
        
        client = AsyncOpenAI()
        
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(colored(f"\n=== Starting Iteration {iteration}/{MAX_ITERATIONS} ===\n", "cyan"))
            
            # Update paths for current iteration
            iteration_paths = get_iteration_paths(iteration)
            global OUTPUT_FILES, METRICS_FILES
            OUTPUT_FILES = iteration_paths['solutions']
            METRICS_FILES = iteration_paths['metrics']
            
            challenge_info = read_challenge_info()
            if not challenge_info:
                return

            # Collect results from previous iteration
            previous_results = collect_previous_results(iteration)
            
            # Create solution keys dynamically
            solution_keys = [f"solution_{i+1}" for i in range(HOW_MANY_SOLUTIONS)]
            solution_keys_str = ", ".join([f"'{key}'" for key in solution_keys])

            # Modify prompt based on iteration
            iteration_context = ""
            if previous_results:
                iteration_context = f"""
                Previous Iteration Results:
                The following shows the results from iteration {iteration-1}. 
                Please analyze these results and generate improved solutions:
                
                {json.dumps(previous_results, indent=2)}
                
                Based on these results, please generate new improved solutions that:
                1. Address any issues or limitations found in previous solutions
                2. Build upon successful approaches
                3. Implement more sophisticated techniques where beneficial
                4. Maintain or improve performance metrics
                """

            prompt = f"""
            You are an expert data scientist. Based on the following challenge information, 
            provide {HOW_MANY_SOLUTIONS} different complete Python solutions for the Spaceship Titanic problem.
            This is iteration {iteration} of {MAX_ITERATIONS}.
            Return them as a JSON object with keys: {solution_keys_str}
            
            {iteration_context}
            
            File Structure Requirements:
            - Solutions will be saved in the 'solutions/iteration_{iteration}' directory
            - Metrics should be saved in the 'metrics/iteration_{iteration}' directory
            - Predictions should be saved in 'submissions/iteration_{iteration}' directory
            
            IMPORTANT: Each solution must follow these exact file naming conventions:
            - For solution N (where N is 1 to {HOW_MANY_SOLUTIONS}):
              * Metrics file MUST be named exactly: 'metrics/iteration_{iteration}/metrics_solution_N.json'
              * Submission file MUST be named exactly: 'submissions/iteration_{iteration}/submission_N.csv'
            
            Example for solution 1:
            ```python
            # Correct metrics file path
            metrics_path = f'metrics/iteration_{iteration}/metrics_solution_1.json'
            
            # Correct submission file path
            submission_path = f'submissions/iteration_{iteration}/submission_1.csv'
            
            # Create directories
            import os
            os.makedirs('metrics/iteration_{iteration}', exist_ok=True)
            os.makedirs('submissions/iteration_{iteration}', exist_ok=True)
            
            # Save metrics (example)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=4)
                
            # Save submission (example)
            submission_df.to_csv(submission_path, index=False)
            ```
            
            IMPORTANT: Do not modify these paths or add additional subdirectories.
            The metrics file MUST be named 'metrics_solution_N.json' where N matches your solution number.
            
            Test Prediction Requirements:
            - Each solution must load and make predictions on the test data from '{TEST_FILE}'
            - Save predictions in a CSV file named 'submissions/iteration_{iteration}/submission_X.csv' (where X is the solution number)
            - The submission file must have exactly two columns: 'PassengerId' and 'Transported'
            - The 'Transported' column should contain boolean predictions (True/False)
            - Ensure the submission file matches the original test file row order

            Hardware Requirements:
            - Each solution should check for GPU availability when using PyTorch, XGBoost, or other GPU-capable libraries
            - Include fallback to CPU if GPU is not available
            - Example GPU check for PyTorch:
            ```python
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {{device}}")
            # Move model and tensors to device
            model = model.to(device)
            X_train = X_train.to(device)
            ```

            These solutions should be designed for RAPID PRELIMINARY EVALUATION:
            - Focus on quick execution and fast iteration
            - Use efficient, lightweight approaches
            - Minimize preprocessing complexity initially
            - Choose methods that train quickly
            - Include basic but meaningful feature engineering
            - Balance speed vs performance for initial insights

            Each solution should be a complete, runnable Python script with different approaches.
            Feel free to be creative and innovative in your approaches while keeping runtime minimal.
            For whenever you need use pytorch instead of tensorflow.
            Make sure the code is error free and runs without errors.

            IMPORTANT: Each solution must include code to:
            1. Train the model on the training data
            2. Make predictions on the test data
            3. Save predictions to a properly formatted submission file
            4. Save comprehensive performance metrics as a JSON file

            The metrics should help evaluate and compare solutions, including but not limited to:
            - Core performance metrics appropriate for the chosen approach
            - Training and inference characteristics
            - Model parameters and configuration
            - Important features or patterns discovered
            - Resource usage
            - Any other relevant metrics that showcase the solution's strengths

            The metrics should be detailed enough to:
            1. Understand how well the solution performs
            2. Enable meaningful comparisons between different approaches
            3. Identify areas for potential improvement
            4. Document key decisions and parameters

            Each solution should also include comments about potential improvements 
            or more complex approaches that could be explored after this initial evaluation.

            Include proper imports, data loading, preprocessing, model training, prediction, and submission file generation.

            Challenge Information:
            {challenge_info}
            """

            print(colored(f"Requesting solutions for iteration {iteration}...", "cyan"))
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"}
            )

            save_token_usage(response, iteration)

            solutions_json = json.loads(response.choices[0].message.content)
            solutions = [solutions_json[key] for key in solution_keys]
            
            save_solutions(solutions)
            print(colored(f"Solutions for iteration {iteration} generated successfully!", "green"))
            
            # Run solutions in parallel with error correction
            await run_solutions_parallel()
            
            print(colored(f"\n=== Completed Iteration {iteration}/{MAX_ITERATIONS} ===\n", "green"))

    except Exception as e:
        print(colored(f"Error in main execution: {str(e)}", "red"))

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()