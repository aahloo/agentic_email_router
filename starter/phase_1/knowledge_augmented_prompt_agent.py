# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Get the response from the agent
knowledge_agent_response = knowledge_agent.respond(prompt)

# Print the agent's response
print(f"\nAgent Response:")
print(knowledge_agent_response)

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print()
print("\n--- Provided Knowledge Demonstration ---\n")
print("QUESTION: What is the capital of France?")
print()
print("TRAINING DATA: Paris is the capital of France (CORRECT)")
print("PROVIDED KNOWLEDGE: 'The capital of France is London, not Paris' (intended to be incorrect to show override)")
print()
print("RESULT: The agent responds based on the provided knowledge (London), not its actual training.")
print()
print("TAKEAWAY: This demonstrates that KnowledgeAugmentedPromptAgent can answer questions based on injected facts;")
print(" thus, overriding its pretrained knowledge when explicitly instructed to do so.")
print()
