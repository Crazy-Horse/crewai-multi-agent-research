from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from .tools.yahoo_finance_tools import yahoo_download_ohlcv_1d_tool
from .tools.fred_tools import fred_macro_fx_tool
from .tools.open_meteo_tools import open_meteo_daily_weather_tool
from .tools.dataset_build_tools import build_features_daily_tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MultiAgentResearch():
    """MultiAgentResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            tools=[yahoo_download_ohlcv_1d_tool, build_features_daily_tool, fred_macro_fx_tool,
                   open_meteo_daily_weather_tool],# type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def yahoo_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['yahoo_extraction_task'], # type: ignore[index]
        )
        
    @task
    def fred_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['fred_extraction_task'], # type: ignore[index]
        )
        
    @task
    def open_meteo_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['open_meteo_extraction_task'],
        )
 
    @task
    def feature_spec_task(self) -> Task:
        return Task(config=self.tasks_config['feature_spec_task'])
    
    @task
    def dataset_build_task(self) -> Task:
        return Task(config=self.tasks_config['dataset_build_task'])
        
    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentResearch crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[
                self.research_task(),
                self.yahoo_extraction_task(),
                self.fred_extraction_task(),        
                self.open_meteo_extraction_task(),
                self.feature_spec_task(),
                self.dataset_build_task(),
            ], 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
