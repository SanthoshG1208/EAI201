import asyncio
from browser_use import Agent 
from browser_use.llm import ChatOpenAI  
from dotenv import load_dotenv


load_dotenv()


async def main():
   
    task = """
      to perform the following browser navigation related to the academic topic 'Artificial Intelligence':

    1. Open https://en.wikipedia.org in the browser.
    2. Search for "Artificial Intelligence".
    3. Click on the main article for "Artificial Intelligence".
    4. From there, follow the hyperlink to "Machine Learning".
    5. Then, navigate to the page on "Deep Learning".
    6. Next, go to the linked page on "Neural Networks".
    7. After that, visit the article about "Natural Language Processing".
    8. Finally, navigate to the "Reinforcement Learning" page.

    """

   
    llm = ChatOpenAI(model="gpt-4o")  

    agent = Agent(
        task=task,
        llm=llm,
      
    )

    result = await agent.run()
    print("Task completed. Result:")
    print(result)

if _name_ == "_main_":
    asyncio.run(main())
