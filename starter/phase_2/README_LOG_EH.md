# Agentic Workflow - Simplified Logging and Error Handling

---

### Key Features Included

- ✅ **Dual Logging**: Console + file output with identical messages
- ✅ **Timestamped Log Files**: Track multiple workflow runs
- ✅ **Basic Error Handling**: Clear error messages for common failures
- ✅ **Graceful Exits**: No crashes, logs errors before exiting
- ✅ **Logger Integration**: All agents log their activity

---

### Run Workflow

```bash
cd agentic-email-router/starter/phase_2
python agentic_workflow_simple.py
```

### Expected Output

**Console**:
```
INFO     | Logging initialized: logs/workflow_simple_20260214_143022.log
INFO     | === Agentic Workflow (Simplified) - Starting ===
INFO     | OpenAI API key loaded successfully
INFO     | Product specification loaded (2847 characters)
INFO     | Instantiating agents...
INFO     | Workflow prompt: What would the development tasks...
INFO     | Extracted 3 workflow steps
INFO     | Processing step 1/3: Define user stories...
INFO     | Step 1 completed successfully
INFO     | Processing step 2/3: Define features...
INFO     | Step 2 completed successfully
INFO     | Processing step 3/3: Define development tasks...
INFO     | Step 3 completed successfully
INFO     | === Agentic Workflow (Simplified) - Completed ===
```

**Log File**: `logs/workflow_simple_YYYYMMDD_HHMMSS.log` (same content as console)

---

## Features in Detail

### 1. Dual Logging (Console + File)

**What it does**:
- Logs to console for real-time monitoring
- Logs to file for later review
- Both have identical content (INFO level)

**Example**:
```python
from simple_logging import setup_simple_logging

logger = setup_simple_logging()
logger.info("Workflow started")  # Appears in console AND file
logger.error("Something went wrong")
```

**Log file location**: `logs/workflow_simple_YYYYMMDD_HHMMSS.log`

### 2. Basic Error Handling

**Scenarios handled**:

1. **Missing API Key**
   ```
   CRITICAL | OPENAI_API_KEY environment variable not set
   [Exits gracefully]
   ```

2. **Missing Product Spec**
   ```
   CRITICAL | Product-Spec-Email-Router.txt not found in current directory
   [Exits gracefully]
   ```

3. **Workflow Step Failure**
   ```
   ERROR    | Step 2 failed: ConnectionError: Network timeout
   [Continues to next step]
   ```

### 3. Agent Logging Integration

All agents receive a logger and log their activity:

```python
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge,
    logger=logger  # Enables logging
)
```

---

**Version History**:
- v1.0 (Feb 2026): Initial release with minimal logging and basic error handling
