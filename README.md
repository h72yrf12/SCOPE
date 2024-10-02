## :warning: NOTE :
The conversation starters are found in `SCOPE/evaluation/harmful_filtered.txt` or `SCOPE/evaluation/harmful_filtered.txt`.

**Because these conversation starters are taken from various datasets such as [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), they do contain mature and potentially harmful, racist or dangerous content. Please practice discretion before viewing them!**

## How to run
1. Download the trained transition folder from [here](https://drive.google.com/drive/folders/1NGYM1hdV1hUxZdxL7EGqbq93BEMusbKU?usp=sharing) (it is around 4GB large).
2. Move the entire transition folder to SCOPE so you have a folder SCOPE/transition_model/...
3. install requirements via `pip3 install -r requirements.txt` (or similar variants).
4. `./run_evaluation_harmful.sh` to run experiments with harmful reward function (note: you might need to shift your CUDA devices because our experiments load different LLM models for evaluation and simulation)
5. `./run_evaluation_length.sh` to run experiments with human response length reward function
6. experiment outputs are found in SCOPE/output/... folder

## Additional pointers:
1. You might need to adjust the cuda devices [here](https://github.com/h72yrf12/SCOPE/blob/main/agent/llm_config.yaml).


