# Agentic Email Router - Enhanced with Simple Logging/Error Handling

AI-powered agentic workflow system for automated email routing and product management planning using OpenAI's GPT models.

## Features

### Phase 1: Individual Agents
- Action Planning Agent
- Knowledge Augmented Prompt Agent
- Evaluation Agent
- Routing Agent
- RAG Knowledge Prompt Agent

### Phase 2: Integrated Workflow (with basic logging and error handling)
- Minimal logging infrastructure (`simple_logging.py`)
- Workflow with basic error handling
- Multi-agent orchestration for:
  - Product Manager: User story generation
  - Program Manager: Feature definition
  - Development Engineer: Task breakdown

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aahloo/agentic_email-router.git
   cd agentic_email-router
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

## Usage

### Phase 1: Individual Agent Testing

Test individual agents:
```bash
cd starter/phase_1
python action_planning_agent.py
python routing_agent.py
# ... test other agents
```

### Phase 2: Integrated Workflow 

Run the workflow:
```bash
cd starter/phase_2
python agentic_workflow_simple.py
```

**Expected Output**:
- Console logs showing workflow progress
- Generated user stories, features, and development tasks
- Log file in `logs/workflow_simple_*.log`

## Documentation

- Logging/Error Handling README](starter/phase_2/README_LOG_EH.md) - Detailed guide for the logging and handling features

## Project Structure

```
agentic-email-router/
├── requirements.txt          # Python dependencies
├── .env.example             # API key template
├── README.md                # This file
└── starter/
    ├── phase_1/             # Individual agents
    │   ├── action_planning_agent.py
    │   ├── routing_agent.py
    │   └── workflow_agents/base_agents.py
    └── phase_2/             # Integrated workflow
        ├── agentic_workflow_simple.py  # Main workflow
        ├── simple_logging.py            # Logging utilities
        ├── Product-Spec-Email-Router.txt
        └── README_LOG_EH.md             # Detailed doc
```
