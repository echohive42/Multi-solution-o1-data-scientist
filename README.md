# O1 Data Scientist - Automated ML Solution Generator

⚠️ **WARNING: API COST & CODE AUTO EXECUTION NOTICE** ⚠️
THIS SCRIPT AUTO EXECUTES AI GENERATED CODE IN YOUR MACHINE WITHOUT A SANDBOX. USE IT WITH CAUTION.

This tool can consume a significant number of tokens due to multiple solutions and parallel error correcting
Please monitor your token usage carefully and adjust the parameters (MAX_ITERATIONS, HOW_MANY_SOLUTIONS) according to your budget.

## Overview
This tool automates the process of generating, testing, and iteratively improving machine learning solutions for the Spaceship Titanic challenge using O1 AI models. It generates multiple solutions with different approaches, automatically fixes errors, and tracks performance metrics across iterations.

## Features
- 🔄 Iterative solution improvement
- 🛠️ Automatic error correction
- 📊 Comprehensive metrics tracking
- 💾 Organized file structure
- 🔍 GPU detection and utilization
- 📈 Token usage tracking
- ⏱️ Parallel solution execution

  ## 🎥 Watch How It's Built!

**[Watch the complete build process on Patreon](https://www.patreon.com/posts/multi-solution-120281912?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)**
See exactly how this automation system was created step by step, with detailed explanations and insights into the development process.

## ❤️ Support & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! [Join my AI community](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools & platforms (up to $180 value)

## Directory Structure
```
project_root/
├── solutions/
│   ├── iteration_1/
│   │   ├── solution_1.py
│   │   ├── solution_2.py
│   │   └── solution_3.py
│   ├── iteration_2/
│   └── iteration_3/
├── metrics/
│   ├── iteration_1/
│   │   ├── metrics_solution_1.json
│   │   ├── metrics_solution_2.json
│   │   └── metrics_solution_3.json
│   ├── iteration_2/
│   └── iteration_3/
├── submissions/
│   ├── iteration_1/
│   ├── iteration_2/
│   └── iteration_3/
├── token_usage.json
├── additional_info.txt
├── train.csv
├── test.csv
└── o1_data_scientist.py
```

## Requirements
```
openai
termcolor
pandas
numpy
scikit-learn
xgboost
```

## Configuration
Key parameters in `o1_data_scientist.py`:
```python
HOW_MANY_SOLUTIONS = 3      # Number of solutions per iteration
MAX_ITERATIONS = 3          # Number of improvement iterations
MAX_RUNTIME_MINUTES = 5     # Maximum runtime per solution
MAX_ERROR_CORRECTION_ATTEMPTS = 5  # Max attempts to fix errors
REASONING_EFFORT = "high"   # O1 reasoning effort level
SAMPLE_ROWS = 30           # Number of dataset rows to include in prompt
```

## Usage
1. Ensure you have the required CSV files:
   - `train.csv`: Training data
   - `test.csv`: Test data for predictions

2. Create `additional_info.txt` with problem description and requirements

3. Run the script:
```bash
python o1_data_scientist.py
```

## Output
- **Solutions**: Complete Python scripts with different ML approaches
- **Metrics**: Performance metrics and model details in JSON format
- **Submissions**: Prediction files for each solution
- **Token Usage**: Detailed token consumption tracking

## Features in Detail

### Iterative Improvement
- Each iteration builds upon insights from previous solutions
- Metrics and performance data guide improvements
- Solutions evolve in sophistication across iterations

### Error Correction
- Automatic detection and fixing of runtime errors
- Uses O1-mini model for quick fixes
- Multiple correction attempts with different approaches

### GPU Utilization
- Automatic GPU detection for supported libraries
- Fallback to CPU when GPU unavailable
- Optimized resource usage

### Metrics Tracking
- Model performance metrics
- Training and inference times
- Resource usage statistics
- Feature importance
- Validation scores

### Token Usage Monitoring
- Per-iteration token counting
- Cumulative usage tracking
- Detailed token type breakdown
- Timestamp logging

## Best Practices
1. Start with small values for HOW_MANY_SOLUTIONS and MAX_ITERATIONS
2. Monitor token usage in token_usage.json
3. Review metrics between iterations
4. Adjust runtime limits based on solution complexity
5. Keep additional_info.txt concise but informative

## Limitations
- Token consumption can be high
- Runtime dependent on model complexity
- Error correction may not fix all issues
- GPU availability affects performance

## Contributing
Feel free to submit issues and enhancement requests!

## License
[Your chosen license] 
