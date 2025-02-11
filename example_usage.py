from ai_support_agent import AIDebugAgent

def main():
    # Initialize the agent
    agent = AIDebugAgent(
        knowledge_base_path='knowledge_base.json'
    )
    
    try:
        # Start debugging a specific issue
        agent.debug_issue('high_cpu_usage')
    except KeyboardInterrupt:
        print("\nDebugging session interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 