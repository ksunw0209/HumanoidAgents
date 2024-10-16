import os
import json
from openai import OpenAI
import re

client = OpenAI()

def load_agent_backgrounds(folder_path):
    backgrounds = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                agent_name = filename.split('.')[0]
                background = json.load(file)
                # Remove "example_day_plan" if it exists
                background.pop('example_day_plan', None)
                backgrounds[agent_name] = background
    return backgrounds

def get_believability_score_and_summary(state, previous_summary, agent_backgrounds):
    system_prompt = f"""
    Evaluate the believability of 3 LLM-based agents by determining how human-like they are. This should be based on their background information, current locations, actions, and memory (past actions from previous user prompts). Provide a score along with reasons for the believability assessment and a summary of the current and past events.

    Previous summary: {previous_summary}

    Agent backgrounds:
    {json.dumps(agent_backgrounds, indent=2)}

    # Steps
    
    1. **Analyze Background Information**: Review the provided background information to understand the context and characteristics of each agent.
    2. **Review Current Location and Actions**: Consider how well the agents' current settings and actions align with typical human behavior or expectations for someone with their background.
    3. **Assess Memory**: Analyze the continuity between the previous summary and current actions to determine if the agents demonstrate human-like continuity and reasoning.
    4. **Assign Score**: Based on the analysis, assign a score that reflects the agents' believability as humans, explaining the reasoning behind the score.
    5. **Generate Summary**: Provide a brief summary of the current and past events for each agent, focusing on their actions and any significant interactions. This summary should serve as a reference memory for future evaluations.
    
    # Output Format
    
    Provide the output as a JSON object with the following fields:
    - `score`: A number between 0 and 10, where 10 indicates high believability.
    - `reasons`: A brief explanation for the assigned score.
    - `summary`: A concise summary of the current and past events for each agent, incorporating relevant past information. Structure this as a dictionary with agent names as keys.
    
    Example:
    ```json
    {{
      "score": 8,
      "reasons": "The agents' actions are consistent with their backgrounds and previous behaviors. There's good continuity in their interactions, though occasional responses seem slightly exaggerated.",
      "summary": {{
        "Sheldon": "Continued work on his theoretical physics problem at Caltech, had lunch with Leonard, and engaged in a debate about Star Trek with Raj.",
        "Leonard": "Conducted experiments in the lab, had lunch with Sheldon, and helped Penny rehearse lines for her audition.",
        "Penny": "Went to an audition for a local theater production, had a shift at the Cheesecake Factory, and practiced lines with Leonard's help."
      }}
    }}
    ```
    
    # Notes
    
    - Consider context from the background information, previous summary, and current scenario thoroughly.
    - Assess consistency in behavior, logical reasoning, and memory recall.
    - Scores closer to 10 indicate high believability, while scores closer to 0 indicate lower human-like attributes.
    - Ensure the summary for each agent focuses on their specific actions and interactions, providing a clear reference for future evaluations.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(state)}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    response_text = response.choices[0].message.content
    try:
        # First, try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        try:
            json_str = response_text.split('```json', 1)[1].split('```', 1)[0]
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            print(f"Error decoding JSON. Raw response: {response_text}")
            return None

def extract_time(filename):
    match = re.search(r'_(\d{2})h', filename)
    if match:
        return int(match.group(1))
    return None

def process_files_chronologically(folder_path, agent_backgrounds):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Filter and sort files based on the specified time pattern
    valid_times = [f"{i:02d}h00" for i in range(6, 24)]  # 06h00 to 23h00
    valid_files = [f for f in files if any(time in f for time in valid_times)]
    sorted_files = sorted(valid_files, key=lambda x: extract_time(x))

    total_score = 0
    file_count = 0
    previous_summary = {"Sheldon": "", "Leonard": "", "Penny": ""}

    for filename in sorted_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            state = json.load(file)
            result = get_believability_score_and_summary(state, previous_summary, agent_backgrounds)
            
            if result:
                total_score += result['score']
                file_count += 1
                previous_summary = result['summary']
                
                print(f"File: {filename}")
                print(f"Score: {result['score']}")
                print(f"Reasons: {result['reasons']}")
                print("Summary:")
                for agent, summary in result['summary'].items():
                    print(f"  {agent}: {summary}")
                print("---")
            else:
                print(f"Failed to process file: {filename}")

    if file_count == 0:
        return None

    average_score = total_score / file_count
    return average_score, previous_summary

# Load agent backgrounds
backgrounds_folder = '../specific_agents'
agent_backgrounds = load_agent_backgrounds(backgrounds_folder)

# Process files
folder_path = '../generations/big_bang_theory_llama'  # Replace with your folder path
result = process_files_chronologically(folder_path, agent_backgrounds)

if result:
    average_score, final_summary = result
    print(f"The average believability score for all agents is: {average_score}")
    print("Final summary of events:")
    for agent, summary in final_summary.items():
        print(f"  {agent}: {summary}")
else:
    print("No valid JSON files found or failed to calculate scores.")