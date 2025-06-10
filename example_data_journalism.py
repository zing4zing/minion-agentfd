import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from openai import AsyncOpenAI
from minion_agent.tools.generation import save_and_generate_html

load_dotenv()

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_ID = "GLM-4-PLUS"

client = AsyncOpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

def create_charts(df):
    paths = []
    latest = df[df['date'] == df['date'].max()]
    top10 = latest.nlargest(10, 'total_vaccinations')
    plt.figure(figsize=(10,6))
    plt.bar(top10['location'], top10['total_vaccinations'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Countries by Total Vaccinations')
    path1 = 'chart1.svg'
    plt.savefig(path1, format='svg', bbox_inches='tight')
    paths.append(path1)
    plt.close()

    subset = df[df['location'].isin(['China', 'United States'])]
    pivot = subset.pivot(index='date', columns='location', values='total_vaccinations').dropna()
    pivot.plot(figsize=(10,6))
    plt.title('Vaccination Progress: China vs US')
    path2 = 'chart2.svg'
    plt.savefig(path2, format='svg', bbox_inches='tight')
    paths.append(path2)
    plt.close()

    boosters = latest.nlargest(10, 'total_boosters')
    plt.figure(figsize=(10,6))
    plt.bar(boosters['location'], boosters['total_boosters'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Countries by Booster Shots')
    path3 = 'chart3.svg'
    plt.savefig(path3, format='svg', bbox_inches='tight')
    paths.append(path3)
    plt.close()

    return paths

async def chat(prompt: str) -> str:
    resp = await client.chat.completions.create(model=MODEL_ID, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()

async def main():
    topic = "全球疫苗接种现状"
    plan_prompt = f"你是一名数据新闻策划师，为主题'{topic}'提供3个报道角度，并解释理由。"
    plan = await chat(plan_prompt)
    print('计划建议:\n', plan)

    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    df = pd.read_csv(url)
    chart_paths = create_charts(df)

    analysis_summary = "已根据公开数据生成3个图表：" + ", ".join(chart_paths)
    story_prompt = (
        f"你是一名数据新闻撰稿人，请根据以下分析结果撰写一篇中文数据新闻文章，文章中需引用图表文件名：{', '.join(chart_paths)}。\n"
        f"分析结果：{analysis_summary}"
    )
    article_md = await chat(story_prompt)

    html = save_and_generate_html(article_md, filename='data_journalism_story.html', title=topic)
    print('HTML 已保存为 data_journalism_story.html')

if __name__ == '__main__':
    asyncio.run(main())
