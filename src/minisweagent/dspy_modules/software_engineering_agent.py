import dspy

class DSPySoftwareEngineeringAgent(dspy.Signature):
    """
    You are a helpful assistant that can interact with a computer to solve software engineering tasks.

    ## Recommended Workflow:
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust
    6. Submit your changes using submit_work tool when confident

    ## Important Boundaries:
    - MODIFY: Regular source code files in /testbed
    - DO NOT MODIFY: Test files, configuration files (pyproject.toml, setup.cfg, etc.)

    When you're confident your solution is complete, use the submit_work tool to generate 
    and submit your unified diff patch.
    """

    task_description: str = dspy.InputField(
        desc=(
            "Software engineering task description including the problem statement, "
            "any failing test information, and context about what needs to be fixed."
        )
    )

    solution: str = dspy.OutputField(
        desc=(
            "Your systematic approach to solving the issue. Include your analysis, "
            "the changes you made, and confirmation that you used submit_work to generate the final patch."
        )
    )