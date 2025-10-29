from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Define the state structure
class GraphState(TypedDict):
    domain: str
    topics: List[Dict[str, any]]
    critic_feedback: str
    iteration: int
    max_iterations: int
    final_output: List[Dict[str, any]]

# Initialize the LLM with Ollama granite3.3:8b
llm = init_chat_model(
    model="granite3.3:8b",
    model_provider="ollama",
    temperature=0.7
)

def actor_generate(state: GraphState) -> GraphState:
    """Actor node: Generates topics and subtopics"""
    domain = state["domain"]
    iteration = state.get("iteration", 0)
    
    if iteration == 0:
        # Initial generation
        prompt = f"""Generate a comprehensive list of 3-4 main topics related to '{domain}'.
For each topic, provide 3-4 relevant subtopics.

Format your response as a structured list:
Topic 1: [Topic Name]
- Subtopic 1.1: [Name]
- Subtopic 1.2: [Name]
...

Be specific, ensure topics are diverse and well-organized, and cover both foundational and advanced aspects."""
        
        messages = [
            SystemMessage(content="You are an expert content organizer and topic generator."),
            HumanMessage(content=prompt)
        ]
    else:
        # Add NEW topics based on critic feedback
        current_topics = state["topics"]
        feedback = state["critic_feedback"]
        
        prompt = f"""Based on the critic's feedback, ADD 2-3 NEW main topics (with 3-4 subtopics each) that complement the existing topics.

EXISTING TOPICS:
{format_topics(current_topics)}

CRITIC FEEDBACK:
{feedback}

Generate NEW topics that:
1. Address gaps mentioned in the feedback
2. Cover different aspects not yet explored
3. Complement but don't duplicate existing topics

Format as:
Topic X: [New Topic Name]
- Subtopic X.1: [Name]
- Subtopic X.2: [Name]
...

IMPORTANT: Only provide the NEW topics, not the existing ones."""
        
        messages = [
            SystemMessage(content="You are an expert content organizer. Generate new topics based on feedback."),
            HumanMessage(content=prompt)
        ]
    
    response = llm.invoke(messages)
    new_topics = parse_topics(response.content)
    
    # Merge new topics with existing ones
    if iteration > 0:
        all_topics = state["topics"] + new_topics
    else:
        all_topics = new_topics
    
    # Print all topics and subtopics after each iteration
    new_iteration = iteration + 1
    print(f"\n{'='*60}")
    print(f"ITERATION {new_iteration} - ALL TOPICS AND SUBTOPICS")
    print(f"{'='*60}")
    print(format_topics(all_topics))
    print(f"Total topics: {len(all_topics)}")
    if iteration > 0:
        print(f"New topics added this iteration: {len(new_topics)}")
    print(f"{'='*60}\n")
    
    return {
        **state,
        "topics": all_topics,
        "iteration": new_iteration
    }

def critic_evaluate(state: GraphState) -> GraphState:
    """Critic node: Evaluates the topics and provides feedback"""
    topics = state["topics"]
    iteration = state["iteration"]
    num_topics = len(topics)
    
    prompt = f"""Evaluate the following {num_topics} topics and subtopics for quality, relevance, and completeness:

{format_topics(topics)}

Provide constructive feedback on:
1. What important areas or perspectives are MISSING that should be added?
2. What gaps exist in the current coverage?
3. What complementary topics would enhance the collection?
4. Are there any emerging trends or advanced concepts not covered?
5. Balance between foundational and advanced topics
6. Any redundancies or overlaps that need attention

Focus especially on suggesting NEW topic areas that would complement the existing ones.
Be specific and actionable in your suggestions."""
    
    messages = [
        SystemMessage(content="You are a critical reviewer who identifies gaps and suggests new topics to add."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        **state,
        "critic_feedback": response.content
    }

def should_continue(state: GraphState) -> str:
    """Decide whether to continue iterating or end"""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    
    if iteration >= max_iterations:
        return "finalize"
    else:
        return "continue"

def finalize_output(state: GraphState) -> GraphState:
    """Finalize the output"""
    return {
        **state,
        "final_output": state["topics"]
    }

# Helper functions
def parse_topics(text: str) -> List[Dict[str, any]]:
    """Parse LLM output into structured format"""
    topics = []
    current_topic = None
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('Topic'):
            if current_topic:
                topics.append(current_topic)
            # Extract topic name
            topic_name = line.split(':', 1)[1].strip() if ':' in line else line
            current_topic = {
                "topic": topic_name,
                "subtopics": []
            }
        elif line.startswith('-') or line.startswith('•'):
            if current_topic:
                subtopic = line.lstrip('-•').strip()
                if ':' in subtopic:
                    subtopic = subtopic.split(':', 1)[1].strip()
                current_topic["subtopics"].append(subtopic)
    
    if current_topic:
        topics.append(current_topic)
    
    return topics

def format_topics(topics: List[Dict[str, any]]) -> str:
    """Format topics for display"""
    output = []
    for i, topic in enumerate(topics, 1):
        output.append(f"Topic {i}: {topic['topic']}")
        for j, subtopic in enumerate(topic['subtopics'], 1):
            output.append(f"  - Subtopic {i}.{j}: {subtopic}")
    return '\n'.join(output)

# Build the graph
def create_actor_critic_graph():
    """Create and return the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("actor", actor_generate)
    workflow.add_node("critic", critic_evaluate)
    workflow.add_node("finalize", finalize_output)
    
    # Add edges
    workflow.set_entry_point("actor")
    workflow.add_edge("actor", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue": "actor",
            "finalize": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    # Create the graph
    app = create_actor_critic_graph()
    
    # Initial state
    initial_state = {
        "domain": "Machine Learning and Deep Learning",
        "topics": [],
        "critic_feedback": "",
        "iteration": 0,
        "max_iterations": 3,  # Actor will refine twice based on critic feedback
        "final_output": []
    }
    
    # Run the pipeline
    print("Starting Actor-Critic Pipeline with Ollama Granite3.3:8b...\n")
    print(f"Domain: {initial_state['domain']}")
    print(f"Max Iterations: {initial_state['max_iterations']}")
    print("Note: Each iteration will ADD new topics based on critic feedback\n")
    
    result = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL TOPICS AND SUBTOPICS")
    print(f"Total Topics Generated: {len(result['final_output'])}")
    print("=" * 60)
    print(format_topics(result["final_output"]))
    print("\n" + "=" * 60)
    print(f"Completed after {result['iteration']} iterations")
    print("=" * 60)
    
    # Optional: Display the final critic feedback
    if result.get("critic_feedback"):
        print("\n" + "=" * 60)
        print("FINAL CRITIC FEEDBACK")
        print("=" * 60)
        print(result["critic_feedback"])