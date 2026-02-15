# agentic_workflow_simple.py
# Simplified workflow with minimal logging and basic error handling

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os
import sys
from dotenv import load_dotenv
from simple_logging import setup_simple_logging

# ============================================================================
# SETUP: Initialize logging
# ============================================================================

logger = setup_simple_logging()
logger.info("=== Agentic Workflow (with logging and error ) - Starting ===")

# ============================================================================
# CONFIGURATION: Load API key and product spec with error handling
# ============================================================================

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
# Load OpenAI API key
try:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.critical("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

    logger.info("OpenAI API key loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load configuration: {type(e).__name__}: {str(e)}")
    sys.exit(1)

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
# Load product specification
try:
    with open("Product-Spec-Email-Router.txt", "r") as f:
        product_spec = f.read()
    logger.info(f"Product specification loaded ({len(product_spec)} characters)")
except FileNotFoundError:
    logger.critical("Product-Spec-Email-Router.txt not found in current directory")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Failed to load product spec: {type(e).__name__}: {str(e)}")
    sys.exit(1)

# ============================================================================
# AGENT INSTANTIATION: Create all agents with logging enabled
# ============================================================================

logger.info("Instantiating agents...")

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning,
    logger=logger
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    f"{product_spec}"
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    logger=logger
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10,
    logger=logger
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
    logger=logger
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10,
    logger=logger
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
    logger=logger
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10,
    logger=logger
)

# ============================================================================
# SUPPORT FUNCTIONS: Route queries to appropriate agents
# ============================================================================

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

def product_manager_support_function(query):
    """Process query through Product Manager knowledge and evaluation agents."""
    result = product_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def program_manager_support_function(query):
    """Process query through Program Manager knowledge and evaluation agents."""
    result = program_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def development_engineer_support_function(query):
    """Process query through Development Engineer knowledge and evaluation agents."""
    result = development_engineer_evaluation_agent.evaluate(query)
    return result['final_response']

# ============================================================================
# ROUTING AGENT: Configure semantic routing
# ============================================================================

# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.

routing_agent = RoutingAgent(openai_api_key, {}, logger=logger)
agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features by grouping user stories. Does not define user stories or tasks",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining development tasks for implementing user stories. Does not define user stories or features",
        "func": lambda x: development_engineer_support_function(x)
    }
]
routing_agent.agents = agents

# ============================================================================
# WORKFLOW EXECUTION: Run multi-agent workflow with basic error handling
# ============================================================================

print("\n" + "="*80)
print("*** Workflow execution started ***")
print("="*80 + "\n")

# Workflow Prompt
workflow_prompt = (
    "Create a comprehensive project plan for the Email Router product that includes:\n"
    "1. User Stories - Define user stories for all key user personas (email recipients, support teams, product managers, admins)\n"
    "2. Product Features - Group related user stories into product features\n"
    "3. Development Tasks - Break down each feature into specific development tasks\n"
    "Generate all three components for a complete project plan."
)
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")
logger.info(f"Workflow prompt: {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).

# Step 1: Extract workflow steps from the prompt
try:
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    logger.info(f"Extracted {len(workflow_steps)} workflow steps")
    print(f"\nExtracted workflow steps:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"  {i}. {step}")
except Exception as e:
    logger.error(f"Failed to extract workflow steps: {type(e).__name__}: {str(e)}")
    print(f"\n❌ ERROR: Failed to extract workflow steps: {str(e)}")
    sys.exit(1)

# Step 2: Initialize completed_steps list
completed_steps = []

# Step 3: Execute each workflow step
print("\n" + "="*80)
print("EXECUTING WORKFLOW STEPS")
print("="*80)

for i, step in enumerate(workflow_steps, 1):
    print(f"\n--- Step {i}/{len(workflow_steps)} ---")
    print(f"Processing: {step}")
    print()

    logger.info(f"Processing step {i}/{len(workflow_steps)}: {step[:60]}...")

    try:
        # Route the step to the appropriate agent
        result = routing_agent.route(step)

        # Store the result
        completed_steps.append(result)
        logger.info(f"Step {i} completed successfully")

        print(f"\nResult:")
        print(result)
        print("\n" + "-"*80)

    except Exception as e:
        logger.error(f"Step {i} failed: {type(e).__name__}: {str(e)}")
        error_msg = f"❌ ERROR: Step {i} failed: {str(e)}"
        print(f"\n{error_msg}")
        # Continue to next step instead of failing entirely
        completed_steps.append(error_msg)
        print("\n" + "-"*80)

# Step 4: Print final workflow output
print("\n" + "="*80)
print("FINAL WORKFLOW OUTPUT")
print("="*80)
print("\nComprehensive Project Plan for Email Router")
print("="*80)

# Organize completed steps by type
user_stories = []
features = []
tasks = []

for i, step in enumerate(completed_steps):
    step_str = str(step).lower()
    # Classify based on content structure
    if "as a" in step_str and "i want" in step_str and "so that" in step_str:
        user_stories.append(step)
    elif "feature name:" in step_str or "key functionality:" in step_str:
        features.append(step)
    elif "task id:" in step_str or "acceptance criteria:" in step_str:
        tasks.append(step)
    else:
        # Fallback: try to infer from position or add to appropriate section
        if i < len(completed_steps) / 3:
            user_stories.append(step)
        elif i < 2 * len(completed_steps) / 3:
            features.append(step)
        else:
            tasks.append(step)

# Print User Stories Section
if user_stories:
    print("\n" + "="*80)
    print("USER STORIES")
    print("="*80)
    for story in user_stories:
        print(f"\n{story}")
        print("-" * 80)

# Print Features Section
if features:
    print("\n" + "="*80)
    print("PRODUCT FEATURES")
    print("="*80)
    for feature in features:
        print(f"\n{feature}")
        print("-" * 80)

# Print Tasks Section
if tasks:
    print("\n" + "="*80)
    print("DEVELOPMENT TASKS")
    print("="*80)
    for task in tasks:
        print(f"\n{task}")
        print("-" * 80)

print("\n*** Workflow execution completed ***\n")
print(f"Generated: {len(user_stories)} user stories, {len(features)} features, {len(tasks)} tasks")

if not completed_steps:
    print("\n❌ No steps were completed successfully.")

# ============================================================================
# COMPLETION: Log completion message
# ============================================================================

logger.info("=== Agentic Workflow (with logging and error handling) - Completed ===")
print("\n*** Workflow execution completed ***")
print(f"\nLog file saved in logs/ directory (check latest workflow_simple_*.log file)")
print()
