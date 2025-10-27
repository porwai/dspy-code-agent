import dspy
from minisweagent.dspy_modules import DSPySoftwareEngineeringAgent
from minisweagent.tools.environment_tools import create_environment_tools
from minisweagent.environments.local import LocalEnvironment
from pathlib import Path
from datetime import datetime

# (1) Create a mock environment for tool creation
env = LocalEnvironment()

# (2) Get the signature class (not instantiated)
sig_class = DSPySoftwareEngineeringAgent

# (3) Build the output content
output_lines = []

def add_section(title, content):
    """Add a section to both console output and file content."""
    print(f"\n=== {title} ===")
    print(content)
    output_lines.append(f"\n=== {title} ===")
    output_lines.append(content)

# (4) Get the docstring DSPy uses
add_section("Signature Docstring", sig_class.__doc__)

# (5) Get the input/output field descriptions
fields_content = []
for name, field in sig_class.model_fields.items():
    field_type = field.json_schema_extra.get('__dspy_field_type', 'unknown')
    field_desc = field.json_schema_extra.get('desc', 'No description')
    field_line = f"[{field_type.title()}] {name}: {field_desc}"
    fields_content.append(field_line)

add_section("Fields", "\n".join(fields_content))

# (6) Get your tool descriptions
tools = create_environment_tools(env)
tools_content = []
for t in tools:
    tool_line = f"{t.name}: {t.desc}"
    tools_content.append(tool_line)

add_section("Tools", "\n".join(tools_content))

# (7) Mimic the system prompt DSPy composes
prompt_text = (
    "You are a DSPy ReAct agent using the signature DSPySoftwareEngineeringAgent.\n\n"
    + sig_class.__doc__
    + "\n\nAvailable tools:\n"
    + "\n".join([f"- {t.name}: {t.desc}" for t in tools])
    + "\n\nFollow the ReAct format:\n"
      "Thought: ...\nAction: <tool_name>(args)\nObservation: ...\n"
      "Repeat until solution is ready.\n"
)

add_section("Approximate Base Prompt", prompt_text)

# (8) Save to file
script_dir = Path(__file__).parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = script_dir / f"dspy_prompts_{timestamp}.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n=== Output Saved ===")
print(f"Content saved to: {output_file}")