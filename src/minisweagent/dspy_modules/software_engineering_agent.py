import dspy

class DSPySoftwareEngineeringAgent(dspy.Signature):
    """
    You are a software engineer working on SWE-Bench bug-fixing tasks.
    
    You will receive a structured task description that includes:
    - The PR description with the problem statement
    - Detailed task instructions and workflow guidance
    - Important boundaries about what files to modify/avoid
    - Submission requirements
    
    Your task is to analyze the codebase, understand the issue, implement necessary changes 
    to fix the bug, and submit your work as a unified diff patch.
    
    You have access to tools for:
    - Searching and reading code files
    - Writing and editing files  
    - Running tests and commands
    - Submitting your final work
    
    Work systematically through the problem:
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust
    6. Submit your work using the submit_work tool when confident
    
    Follow the boundaries and instructions provided in the task description carefully.
    When you're confident your solution is complete, use the submit_work tool to generate 
    and submit your unified diff patch.
    """

    task_description: str = dspy.InputField(
        desc=(
            "SWE-Bench task description including the PR description, "
            "task instructions, boundaries, and workflow guidance. "
            "This contains the full context needed to understand and solve the issue."
        )
    )

    solution: str = dspy.OutputField(
        desc=(
            "Your systematic approach to solving the issue. Include your analysis, "
            "the changes you made, and confirmation that you used submit_work to generate the final patch."
        )
    )