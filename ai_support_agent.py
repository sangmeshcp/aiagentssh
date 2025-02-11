from crewai import Agent, Task, Crew
from typing import Dict, List, Optional
import json
import logging
import subprocess
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from pydantic import BaseModel, Field, ConfigDict

@dataclass
class DebugStep:
    command: str
    description: str
    expected_output: str
    remediation: Dict[str, str]

class TerminalCommand(BaseModel):
    """Schema for terminal command execution using any available terminal, showing live interactions to the user"""
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True
    )
    command: str = Field(..., description="The command to execute in the terminal")

def execute_terminal_command(command: str) -> str:
    """
    Execute a terminal command and print live output.
    Returns the combined output after the command completes.
    """
    try:
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output = []
        # Read and print output live
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="")  # Shows live interaction to the user
                output.append(line)
        process.wait()
        return "".join(output)
    except Exception as e:
        return f"Error executing command: {str(e)}"

class AIDebugAgent:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.logger = self._setup_logging()
        
        # Using ChatOpenAI with Ollama base URL
        self.llm = ChatOpenAI(
            model="ollama/llama3.2",  # or any model you have in Ollama
            temperature=0.7,
            base_url="http://localhost:11434/v1",  # Ollama API endpoint
            api_key="not-needed"  # Placeholder since Ollama doesn't need an API key
        )
        
        self.terminal_tool = Tool(
            name="terminal_command",
            func=execute_terminal_command,
            description="Execute commands in a terminal and display live output to the user",
            args_schema=TerminalCommand
        )
        self._setup_crew()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('AIDebugAgent')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_knowledge_base(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _setup_crew(self):
        # Create specialized agents with the updated terminal command tool
        self.analyzer_agent = Agent(
            name="Analyzer",
            llm=self.llm,
            tools=[self.terminal_tool],
            role="System output analyzer who interprets command results",
            goal="Analyze system outputs and identify issues",
            backstory="I am an expert system analyzer with deep knowledge of Linux/Unix systems. I specialize in interpreting system outputs and identifying root causes of issues.",
            verbose=True
        )
        
        self.executor_agent = Agent(
            name="Executor",
            llm=self.llm,
            tools=[self.terminal_tool],
            role="Command executor who runs system commands safely",
            goal="Execute system commands and collect outputs, analyze the output and make a fix to achieve the expected output",
            backstory="I am a skilled system administrator with expertise in executing commands safely and efficiently. I understand the implications of each command I run.",
            verbose=True
        )
        
        self.remediation_agent = Agent(
            name="Remediator",
            llm=self.llm,
            tools=[self.terminal_tool],
            role="Problem solver who suggests and implements fixes",
            goal="Provide and implement solutions for identified issues",
            backstory="I am an experienced troubleshooter who specializes in developing and implementing solutions for system issues. I carefully consider the impact of each fix.",
            verbose=True
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[self.analyzer_agent, self.executor_agent, self.remediation_agent],
            tasks=[],
            verbose=True
        )
    
    def ask_user_permission(self, action: str) -> bool:
        while True:
            response = input(f"\nShould I proceed with: {action}? (yes/no): ").lower()
            if response in ['yes', 'y']:
                return True
            if response in ['no', 'n']:
                return False
            print("Please answer with 'yes' or 'no'")
    
    def debug_issue(self, issue_type: str):
        if issue_type not in self.knowledge_base:
            self.logger.error(f"Unknown issue type: {issue_type}")
            return
        
        debug_steps = self.knowledge_base[issue_type]
        
        for step in debug_steps:
            self.logger.info(f"Step: {step['description']}")
            
            if not self.ask_user_permission(step['description']):
                self.logger.info("User chose to skip this step")
                continue
            
            # Create tasks for the crew
            execute_task = Task(
                description=f"Execute command: {step['command']}",
                expected_output="Command execution output",
                agent=self.executor_agent
            )
            
            analyze_task = Task(
                description=(
                    f"Analyze output for: {step['description']}\n"
                    f"Expected output: {step['expected_output']}\n"
                    f"Possible issues: {json.dumps(step['remediation'], indent=2)}"
                ),
                expected_output="Analysis of system state and identified issues",
                agent=self.analyzer_agent
            )
            
            remediate_task = Task(
                description=(
                    "Based on the analysis, implement the most appropriate fix from "
                    f"these options:\n{json.dumps(step['remediation'], indent=2)}"
                ),
                expected_output="Implemented fix and its results",
                agent=self.remediation_agent
            )
            
            # Update crew tasks
            self.crew.tasks = [execute_task, analyze_task, remediate_task]
            
            try:
                # Run the crew
                result = self.crew.kickoff()
                self.logger.info(f"Analysis result: {result}")
            except Exception as e:
                self.logger.error(f"Error during task execution: {str(e)}")
                if not self.ask_user_permission("Continue despite error?"):
                    break
            
            if not self.ask_user_permission("Continue to next step?"):
                self.logger.info("User chose to stop debugging")
                break 