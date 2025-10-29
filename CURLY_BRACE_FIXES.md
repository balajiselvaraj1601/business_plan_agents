# Curly Brace Escaping Fixes

## Issue Summary
Python's `.format()` method treats `{` and `}` as special characters for variable substitution. When prompts contain JSON examples or other literal curly braces, these must be escaped by doubling them (`{{` and `}}`), otherwise a `KeyError` is raised.

## Files Fixed

### 1. `src/prompts/prompt_collection.py`

#### PROMPT_PLANNING_CRITIQUE (Line ~325)
**Issue:** JSON example in `<Instruction>` section had unescaped curly braces
**Fix:** Escaped all curly braces in the JSON example:
- Changed `{` to `{{`
- Changed `}` to `}}`

**Before:**
```python
{
"assessment": "<high-level evaluation considering all criteria>",
"score": <numeric score from 1 to 10>,
...
}
```

**After:**
```python
{{
"assessment": "<high-level evaluation considering all criteria>",
"score": <numeric score from 1 to 10>,
...
}}
```

#### PROMPT_PLANNING (Line ~170)
**Issue:** JSON example in `<Example>` section had unescaped curly braces
**Fix:** Escaped all curly braces in the JSON array examples:
- Changed `{` to `{{`
- Changed `}` to `}}`
- Also escaped `{business}` and `{location}` placeholders within string literals to `{{business}}` and `{{location}}`

**Before:**
```python
[
{
    "topic": "financial analysis",
    "reason": "Financial analysis is critical for launching **{business}** in **{location}**...",
    ...
},
...
]
```

**After:**
```python
[
{{
    "topic": "financial analysis",
    "reason": "Financial analysis is critical for launching **{{business}}** in **{{location}}**...",
    ...
}},
...
]
```

### 2. `src/states/state_collection.py`

#### FeedbackResponse Model (Line ~337)
**Issue:** Model fields had incorrect default values (empty strings for list fields) and missing `populate_by_name` configuration
**Fix:** 
1. Changed default values for list fields from `""` to `[]`
2. Added `model_config = {"populate_by_name": True}` to enable field name usage alongside aliases

**Before:**
```python
class FeedbackResponse(BaseModel):
    ...
    strength_list: List[str] = Field("", ...)
    weakness_list: List[str] = Field("", ...)
```

**After:**
```python
class FeedbackResponse(BaseModel):
    model_config = {"populate_by_name": True}
    ...
    strength_list: List[str] = Field([], ...)
    weakness_list: List[str] = Field([], ...)
```

### 3. `src/planner/smart_planner.py`

#### generate_and_save_business_plan (Line ~260)
**Issue:** LangGraph returns dictionary by default, not PlanningState object
**Fix:** Added safe dictionary access with type checking

**Before:**
```python
result = PlanningResponse(topics=final_state.topics)
```

**After:**
```python
topics = final_state.get("topics", []) if isinstance(final_state, dict) else final_state.topics
result = PlanningResponse(topics=topics)
```

## Verification

All fixes were verified with:
1. ✅ Unit tests (30/30 passing in `test_pydantic_parsers.py`)
2. ✅ Manual prompt formatting tests with actual data
3. ✅ End-to-end smart planner execution
4. ✅ JSON file generation and loading

## Prompts Checked (No Issues Found)

The following prompts were checked and confirmed to NOT have this issue:
- `PROMPT_REPORTING` - No JSON examples with curly braces
- `PROMPT_GENERAL_ROLE` - No JSON examples with curly braces
- `PROMPT_BUSINESS_ANALYSIS_ROLE` - Not used with `.format()`
- `EXPERT_ROUTER_PROMPT` - Uses f-string (evaluated at module load), no JSON examples

## Prevention

To prevent this issue in the future:
1. Always escape curly braces in prompt literal text: `{{` and `}}`
2. Only use single curly braces for `.format()` placeholders: `{variable_name}`
3. Test prompts with `.format()` calls before deployment
4. Consider using f-strings for prompts that don't need runtime formatting

## Code Quality Improvements

### Helper Function Refactoring

#### format_list_as_bullets()
**Purpose:** Eliminate code duplication for formatting lists with bullet points

**Location:** `src/planner/smart_planner.py` (Line ~72)

**Function:**
```python
def format_list_as_bullets(items: List[str]) -> str:
    """
    Format a list of strings as bullet-pointed items.
    
    Args:
        items: List of strings to format
    
    Returns:
        Formatted string with each item prefixed by "- " on a new line
    
    Example:
        >>> format_list_as_bullets(["Item 1", "Item 2"])
        "- Item 1\n- Item 2"
    """
    return "\n".join(f"- {item}" for item in items)
```

**Usage in planner() function:**

Before:
```python
strength_list="\n".join(f"- {item}" for item in state.feedback.strength_list),
weakness_list="\n".join(f"- {item}" for item in state.feedback.weakness_list),
suggestion_list="\n".join(f"- {item}" for item in state.feedback.suggestion_list),
recommendation_list="\n".join(f"- {item}" for item in state.feedback.recommendation_list),
```

After:
```python
strength_list=format_list_as_bullets(state.feedback.strength_list),
weakness_list=format_list_as_bullets(state.feedback.weakness_list),
suggestion_list=format_list_as_bullets(state.feedback.suggestion_list),
recommendation_list=format_list_as_bullets(state.feedback.recommendation_list),
```

**Benefits:**
1. ✅ Eliminates code duplication (DRY principle)
2. ✅ Improves readability and maintainability
3. ✅ Centralized logic for consistent formatting
4. ✅ Reusable across the codebase
5. ✅ Well-documented with docstring and example

**Tests:**
- ✅ Empty list handling
- ✅ Single item formatting
- ✅ Multiple items formatting
- ✅ Integration with planner function
- ✅ All 30 unit tests passing
