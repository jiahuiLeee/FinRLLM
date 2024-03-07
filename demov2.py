# -*- coding: utf-8 -*-
""" 
@Time    : 2024/3/1 10:18
@Author  : Li_Jiahui
@FileName: app.py
@Function:
"""
import random
import csv

from datetime import date, datetime, timedelta
from openai import OpenAI
import gradio as gr
import akshare as ak


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-IkyMvUBTTltHa40zxmdolJxZzrMV5vpYhIes0Lt9s9TacsPM",  # 去 https://github.com/chatanywhere/GPT_API_free 申请一个免费的api key
    base_url="https://api.chatanywhere.tech/v1"
)

# 非流式响应
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答
    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    # print(completion.choices[0].message.content)
    return completion


def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)
    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

def get_credit_data(Type):
    """
    :param Type:
    :return: data_dict
    """
    if Type == "cd":
        data_dict = {}
        label = []
        with open('credit_data/cd2.csv', mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for index, row in enumerate(csv_reader):
                label.append(row['default.payment.next.month'])
                row.pop('ID', None)
                row.pop('default.payment.next.month', None)
                data_dict[index] = row
            random_index = random.randint(0, len(label) + 1)
            customer_data = data_dict[random_index]
            print("label:", label[random_index])
            PAY_str = customer_data["PAY_0"] + "," + customer_data["PAY_2"] + "," + \
                    customer_data["PAY_3"] + "," + customer_data["PAY_4"] + "," + customer_data["PAY_5"] + "," + \
                    customer_data["PAY_6"]
            BILL_AMT_str = customer_data["BILL_AMT1"] + "," + customer_data["BILL_AMT2"] + "," + customer_data["BILL_AMT3"] + "," + \
                        customer_data["BILL_AMT4"] + "," + customer_data["BILL_AMT5"] + "," + customer_data["BILL_AMT6"]
            PAY_AMT_str = customer_data["PAY_AMT1"] + "," + customer_data["PAY_AMT2"] + "," + customer_data["PAY_AMT3"] + "," + \
                        customer_data["PAY_AMT4"] + "," + customer_data["PAY_AMT5"] + "," + customer_data["PAY_AMT6"]
            return customer_data["LIMIT_BAL"], customer_data["SEX"], customer_data["EDUCATION"], customer_data["MARRIAGE"], customer_data["AGE"], PAY_str, BILL_AMT_str, PAY_AMT_str
    elif Type == "ld":
        data_dict = {}
        label = []
        with open('credit_data/ld1.csv', mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for index, row in enumerate(csv_reader):
                label.append(row['BAD'])
                row.pop('BAD', None)
                data_dict[index] = row
            random_index = random.randint(0, len(label) + 1)
            customer_data = data_dict[random_index]
            print("label:", label[random_index])
            return tuple(customer_data.values())
    elif Type == "cf":
        data_dict = {}
        label = []
        with open('credit_data/cf1.csv', mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for index, row in enumerate(csv_reader):
                label.append(row["fraud"])
                row.pop('', None)
                row.pop('cust_id', None)
                row.pop('fraud', None)
                data_dict[index] = row
            random_index = random.randint(0, len(label) + 1)
            customer_data = data_dict[random_index]
            print("label:", label[random_index])
            return tuple(customer_data.values())
    elif Type == "cc":
        data_dict = {}
        label = []
        with open('credit_data/cc2.csv', mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for index, row in enumerate(csv_reader):
                label.append(row['Exited'])
                row.pop('Exited', None)
                row.pop('RowNumber', None)
                row.pop('CustomerId', None)
                row.pop('Surname', None)
                data_dict[index] = row
            random_index = random.randint(0, len(label) + 1)
            customer_data = data_dict[random_index]
            print("label:", label[random_index])
            return tuple(customer_data.values())
    else:
        raise ValueError("Type should be 'cd', 'cf', or 'cc'")


def get_row_prompt(Type, row_data_dict):
    """

    :return:
    """
    prompt_start = "构建一个简明的客户档案描述，包括以下所有信息:\n\n"
    prompt_end = "以下是对这些特征的含义的描述\n"

    if Type == "cd":
        prompt_end = prompt_end + \
                    "LIMIT_BAL: 以新台币计的信贷金额(包括个人及家庭/补充信贷)\n" \
                    "SEX: 性别(1=男性，2=女性)\n" \
                    "EDUCATION: 个人的教育水平(1=研究生院，2=大学，3=高中，4=其他，5=未知)\n" \
                    "MARRIAGE: 个人的婚姻状况(1=已婚，2=单身，3=其他)\n" \
                    "AGE: 以年为单位的个人年龄\n" \
                    "PAY_0 to PAY_6: 4月至9月的还款情况，-1=按时还款，1=迟还款一个月，2=迟还款两个月，…， 8=延迟8个月还款，9=延迟9个月或更长时间还款\n" \
                    "BILL_AMT1 to BILL_AMT6: 4月至9月的账单金额（新台币）\n" \
                    "PAY_AMT1 to PAY_AMT6: 4月至9月的上期付款金额（新台币）"
    elif Type == "ld":
        prompt_end = prompt_end + \
                    "LOAN: 借款人所借的金额\n" \
                    "MORTDUE: 借款人的抵押贷款所欠的金额\n" \
                    "VALUE: 借款人财产的评估价值\n" \
                    "REASON: 贷款的原因，如债务合并，家庭改善，或其他目的\n" \
                    "JOB: 表示借款人的职业或就业状态\n" \
                    "YOJ: 借款人受雇的年数\n" \
                    "DEROG: 借款人的信用报告的贬损评论的数量\n" \
                    "DELINQ: 借款人拖欠付款的次数\n" \
                    "CLAGE: 借款人最早信用额度的年龄，即从借款人开立其第一个信用账户到现在的时间长度\n" \
                    "NINQ: 借款人在过去六个月内开设的信贷额度\n" \
                    "CLNO: 借款人拥有的信用额度总数\n" \
                    "DEBTINC: 借款人的债务收入比，其计算方法是将借款人每月的总债务支付除以他们的月收入"
    elif Type == "cf":
        prompt_end = prompt_end + \
                    "ACTIVATE_DATE: 信用卡的激活日期\n"\
                    "LAST_PAYMENT_DATE: 最近一次使用信用卡支付的日期\n"\
                    "BALANCE: 账户中剩余的可用于购买的余额\n"\
                    "BALANCEFREQUENCY: 余额更新的频率，得分在0到1之间（1=频繁更新，0=不频繁更新）\n"\
                    "PURCHASES: 从账户中进行的购买金额\n"\
                    "ONEOFFPURCHASES: 一次性购买的最大金额\n"\
                    "INSTALLMENTSPURCHASES: 分期购买的金额\n"\
                    "CASHADVANCE: 用户提前获取的现金\n"\
                    "PURCHASESFREQUENCY: 购买的频率，得分在0到1之间（1=频繁购买，0=不频繁购买）\n"\
                    "ONEOFFPURCHASESFREQUENCY: 一次性购买的频率（1=频繁购买，0=不频繁购买）\n"\
                    "PURCHASESINSTALLMENTSFREQUENCY: 分期购买的频率（1=频繁进行，0=不频繁进行）\n"\
                    "CASHADVANCEFREQUENCY: 提前支付现金的频率\n"\
                    "CASHADVANCETRX: 使用“提前支付现金”的交易次数\n"\
                    "PURCHASESTRX: 进行的购买交易次数\n"\
                    "CREDITLIMIT: 用户的信用卡限额\n"\
                    "PAYMENTS: 用户所做的支付金额\n"\
                    "MINIMUM_PAYMENTS: 用户所做的最小支付金额\n"\
                    "PRCFULLPAYMENT: 用户支付的全额支付百分比\n"\
                    "TENURE: 用户信用卡服务的期限\n"
    elif Type == "cc":
        prompt_end = prompt_end + \
                    "CreditScore: 客户的信用评分\n"\
                    "Geography: 客户的地理位置\n"\
                    "Gender: 客户的性别\n"\
                    "Age: 客户年龄\n"\
                    "Tenure: 客户作为银行客户的年数\n"\
                    "Balance: 客户账户余额\n"\
                    "NumOfProducts: 客户通过银行购买的产品数量\n"\
                    "HasCrCard: 表示客户是否有信用卡\n"\
                    "IsActiveMember: 表示客户是否活跃\n"\
                    "EstimatedSalary: 客户的评估收入"
    else:
        raise ValueError("Type should be 'cd', 'ld', 'cf', or 'cc'")

    # 构建请求客户描述的prompt
    data_to_str = ",".join([f"{key}:{value}" for key, value in row_data_dict.items()])
    prompt_ask_descript = prompt_start + data_to_str + '\n\n' + prompt_end

    # 发送请求获取响应
    messages = [{'role': 'user', 'content': prompt_ask_descript}]
    answer = gpt_35_api(messages)

    return prompt_ask_descript, answer.choices[0].message.content


def construct_prompt_bank(Type, customer_profile):
    risk = ""
    if Type == "cd":
        risk = "信用卡违约"
    elif Type == "ld":
        risk = "贷款违约"
    elif Type == "cf":
        risk = "信用卡欺诈"
    elif Type == "cc":
        risk = "客户流失"
    else:
        raise ValueError("Type should be 'cd', 'ld', 'cf', or 'cc'")
    # 构建prompt
    prompt_head = f"你是一位经验丰富的金融风险分析师。你的任务是根据下面的客户描述来分析是否存在{risk}的风险，并分析原因。\n\n"
    prompt_tail = "你的答案格式应该如下:\n\n[预测结果]:\n是或否\n\n[分析结果]:\n…"
    customer_profile = "客户描述 " + customer_profile
    prompt_total = prompt_head + customer_profile + "\n\n" + prompt_tail
    return prompt_total


def predict_analyze(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    answer = gpt_35_api(messages)
    return answer.choices[0].message.content

def operate_cd(bal, sex, edu, marry, age, pay, bill_amt, pay_amt):
    info_dict = {"LIMIT_BAL": bal,
                "SEX": sex,
                "EDUCATION": edu,
                "MARRIAGE": marry,
                "AGE": age,
                "PAY_0 to PAY_6": pay,
                "BILL_AMT1 to BILL_AMT6": bill_amt,
                "PAY_AMT1 to PAY_AMT6": pay_amt
                }
    customer_prompt, customer_profile = get_row_prompt("cd", info_dict)
    print(f">>>>  Customer Prompt  <<<<\n\n{customer_prompt}\n\n>>>>  Customer Profile  <<<<\n\n{customer_profile}")
    print("=" * 30)
    prompt_total = construct_prompt_bank("cd", customer_profile)
    print(prompt_total)
    print("=" * 30)
    answer = predict_analyze(prompt_total)
    print(answer)
    return prompt_total, answer

def operate_ld(loan, mor, val, rea, job, yoj, derog, delinq, clage, ninq, clnq, debtinc):
    info_dict = {"LOAN": loan,
                "MORTDUE": mor,
                "VALUE": val,
                "REASON": rea,
                "JOB": job,
                "YOJ": yoj,
                "DEROG": derog,
                "DELINQ": delinq,
                "CLAGE": clage,
                "NINQ": ninq,
                "CLNQ": clnq,
                "DEBTINC": debtinc
                }
    customer_prompt, customer_profile = get_row_prompt("ld", info_dict)
    print(f">>>>  Customer Prompt  <<<<\n\n{customer_prompt}\n\n>>>>  Customer Profile  <<<<\n\n{customer_profile}")
    print("=" * 30)
    prompt_total = construct_prompt_bank("ld", customer_profile)
    print(prompt_total)
    print("=" * 30)
    answer = predict_analyze(prompt_total)
    print(answer)
    return prompt_total, answer

def operate_cf(ad, lpd, bal, balf, pur, onepur, inspur, cash, purf, onepurf, inpurf, cashf, cashtrx, purtrx, crel, payment, minipay, fullpay, tenure):
    info_dict = {"ACTIVATE_DATE": ad,
                "LAST_PAYMENT_DATE": lpd,
                "BALANCE": bal,
                "BALANCEFREQUENCY": balf,
                "PURCHASES": pur,
                "ONEOFFPURCHASES": onepur,
                "INSTALLMENTSPURCHASES": inspur,
                "CASHADVANCE": cash,
                "PURCHASESFREQUENCY": purf,
                "ONEOFFPURCHASESFREQUENCY": onepurf,
                "PURCHASESINSTALLMENTSFREQUENCY": inpurf,
                "CASHADVANCEFREQUENCY": cashf,
                "CASHADVANCETRX": cashtrx,
                "PURCHASESTRX": purtrx,
                "CREDITLIMIT": crel,
                "PAYMENTS": payment,
                "MINIMUM_PAYMENTS": minipay,
                "PRCFULLPAYMENT": fullpay,
                "TENURE": tenure
                }
    customer_prompt, customer_profile = get_row_prompt("cf", info_dict)
    print(f">>>>  Customer Prompt  <<<<\n\n{customer_prompt}\n\n>>>>  Customer Profile  <<<<\n\n{customer_profile}")
    print("=" * 30)
    prompt_total = construct_prompt_bank("cf", customer_profile)
    print(prompt_total)
    print("=" * 30)
    answer = predict_analyze(prompt_total)
    print(answer)
    return prompt_total, answer

def operate_cc(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    info_dict = {
                "CreditScore": CreditScore,
                "Geography": Geography,
                "Gender": Gender,
                "Age": Age,
                "Tenure": Tenure,
                "Balance": Balance,
                "NumOfProducts": NumOfProducts,
                "HasCrCard": HasCrCard,
                "IsActiveMember": IsActiveMember,
                "EstimatedSalary": EstimatedSalary
                }
    customer_prompt, customer_profile = get_row_prompt("cc", info_dict)
    print(f">>>>  Customer Prompt  <<<<\n\n{customer_prompt}\n\n>>>>  Customer Profile  <<<<\n\n{customer_profile}")
    print("=" * 30)
    prompt_total = construct_prompt_bank("cc", customer_profile)
    print(prompt_total)
    print("=" * 30)
    answer = predict_analyze(prompt_total)
    print(answer)
    return prompt_total, answer

    """
    市场风险预测
    """
def get_curday():
    """获取当前系统时间
    Returns:
        str: current date
    """
    return date.today().strftime("%Y-%m-%d")

def get_stock_data(symbol, step, startDate):
    endDate = ""
    stock_data = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=startDate,
            end_date=endDate,
            adjust=""
            )
    print(stock_data)

def get_news_data():
    pass

def get_report_data():
    pass

def crawl_metadata(ticker, date, n_weeks):
    # 获取股票数据
    
    # 获取新闻数据
    
    # 获取财报数据
    pass

def construct_prompt_stock(data_descript):
    # 构建prompt
    prompt_head = f"你是一位经验丰富的金融风险分析师。你的任务是根据过去几周的相关新闻和基本财务数据来分析公式股市的存在什么风险，并分析原因。\n\n"
    prompt_tail = "你的答案格式应该如下:\n\n[预测结果]:\n上涨或者下跌\n\n[分析结果]:\n…"
    prompt_total = prompt_head + data_descript + "\n\n" + prompt_tail
    return prompt_total

def operate_stock(ticker, date, n_weeks):
    # 爬取数据
    info, prompt = crawl_metadata(ticker, date, n_weeks)
    # 生成prompt
    prompt_total = construct_prompt_stock(prompt)
    # 输入模型
    answer = predict_analyze(prompt_total)
    # 返回 prompt 和 回答
    return prompt_total, answer

def clean():
    return ("","",0,"","")


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown("金融风险大模型项目")
        gr.Markdown("面向银行的风险预测")
        with gr.Tabs():
            with gr.Tab("信用卡违约风险"):
                with gr.Row():
                    with gr.Column():
                        inputs = [
                            gr.Textbox(label="贷款金额",info="以新台币计的信贷金额(包括个人及家庭/补充信贷)"),
                            gr.Textbox(label="性别",info="性别(1=男性，2=女性)"),
                            gr.Textbox(label="教育水平",info="1=研究生院，2=大学，3=高中，4=其他，5=未知"),
                            gr.Textbox(label="婚否",info="1=已婚，2=单身，3=其他"),
                            gr.Textbox(label="年龄",info="以年为单位的个人年龄"),
                            gr.Textbox(label="延期还款情况",info="4月至9月的还款情况，-1=按时还款，1=迟还款一个月，2=迟还款两个月，…， 8=延迟8个月还款，9=延迟9个月或更长时间还款"),
                            gr.Textbox(label="账单情况",info="4月至9月的账单金额（新台币）"),
                            gr.Textbox(label="还款金额",info="4月至9月的上期付款金额（新台币）")]
                        with gr.Row():
                            generate_btn = gr.Button("生成数据")
                            submit_btn = gr.Button("提交数据")
                    with gr.Column():
                        output1 = gr.Textbox(label="提问模板", lines=5)
                        output2 = gr.Textbox(label="预测分析", lines=5)
                generate_btn.click(fn=lambda:get_credit_data("cd"), inputs=[], outputs=inputs)
                submit_btn.click(fn=operate_cd, inputs=inputs, outputs=[output1, output2])
                
            with gr.Tab("贷款违约风险"): 
                with gr.Row():
                    with gr.Column():
                        inputs = [
                            gr.Textbox(label="贷款金额", info="借款人所借的金额"),
                            gr.Textbox(label="按揭", info="现有按揭的应付金额"),
                            gr.Textbox(label="资产",info="借款人财产的评估价值"),
                            gr.Textbox(label="贷款原因",info="DebtCon=债务整合;HomeImp=房屋改善"),
                            gr.Textbox(label="职业",info=""),
                            gr.Textbox(label="工作年限",info=""),
                            gr.Textbox(label="贬损数量",info="借款人的信用报告的贬损评论的数量"),
                            gr.Textbox(label="拖欠付款的次数",info=""),
                            gr.Textbox(label="借款人开立其第一个信用账户到现在的时间长度",info=""),
                            gr.Textbox(label="借款人在过去六个月内开设的信贷额度",info=""),
                            gr.Textbox(label="借款人拥有的信用额度总数",info=""),
                            gr.Textbox(label="借款人的债务收入比",info="")]
                        with gr.Row():
                            generate_btn = gr.Button("生成数据")
                            submit_btn = gr.Button("提交数据")
                    with gr.Column():
                        output1 = gr.Textbox(label="提问模板", lines=5)
                        output2 = gr.Textbox(label="预测分析", lines=5)

                generate_btn.click(fn=lambda:get_credit_data("ld"), inputs=[], outputs=inputs)
                submit_btn.click(fn=operate_ld, inputs=inputs, outputs=[output1, output2])
            
            with gr.Tab("信用卡欺诈风险"):
                with gr.Row():
                    with gr.Column():
                        inputs = [
                            gr.Textbox(label="信用卡激活日期", info=""),
                            gr.Textbox(label="最近一次使用信用卡日期", info=""),
                            gr.Textbox(label="账户余额",info=""),
                            gr.Textbox(label="余额更新频率",info=""),
                            gr.Textbox(label="已消费金额",info=""),
                            gr.Textbox(label="单次购买最大金额",info=""),
                            gr.Textbox(label="分期购买金额",info=""),
                            gr.Textbox(label="取现金额",info=""),
                            gr.Textbox(label="购买频率",info="1=频繁购买，0=不频繁购买"),
                            gr.Textbox(label="一次性购买频率",info="1=频繁购买，0=不频繁购买"),
                            gr.Textbox(label="分期购买频率",info="1=频繁购买，0=不频繁购买"),
                            gr.Textbox(label="取现频率",info="1=频繁，0=不频繁"),
                            gr.Textbox(label="取现次数",info=""),
                            gr.Textbox(label="交易次数",info=""),
                            gr.Textbox(label="信用卡限额",info=""),
                            gr.Textbox(label="已偿还金额",info=""),
                            gr.Textbox(label="最小偿还金额",info=""),
                            gr.Textbox(label="全额偿还百分百",info=""),
                            gr.Textbox(label="信用卡服务的期限",info=""),]
                        with gr.Row():
                            generate_btn = gr.Button("生成数据")
                            submit_btn = gr.Button("提交数据")
                    with gr.Column():
                        output1 = gr.Textbox(label="提问模板", lines=5)
                        output2 = gr.Textbox(label="预测分析", lines=5)

                generate_btn.click(fn=lambda:get_credit_data("cf"), inputs=[], outputs=inputs)
                submit_btn.click(fn=operate_cf, inputs=inputs, outputs=[output1, output2])
            
            with gr.Tab("客户流失风险"):
                with gr.Row():
                    with gr.Column():
                        inputs = [
                            gr.Textbox(label="信用分数", info=""),
                            gr.Textbox(label="地理位置", info=""),
                            gr.Textbox(label="性别",info=""),
                            gr.Textbox(label="年龄",info=""),
                            gr.Textbox(label="客户年限",info=""),
                            gr.Textbox(label="账户余额",info=""),
                            gr.Textbox(label="购买的产品数量",info=""),
                            gr.Textbox(label="是否有信用卡",info=""),
                            gr.Textbox(label="是否活跃",info=""),
                            gr.Textbox(label="评估收入",info="")]
                        with gr.Row():
                            generate_btn = gr.Button("生成数据")
                            submit_btn = gr.Button("提交数据")
                    with gr.Column():
                        output1 = gr.Textbox(label="提问模板", lines=5)
                        output2 = gr.Textbox(label="预测分析", lines=5)

                generate_btn.click(fn=lambda:get_credit_data("cc"), inputs=[], outputs=inputs)
                submit_btn.click(fn=operate_cc, inputs=inputs, outputs=[output1, output2])
        
        gr.Markdown("面向企业的市场风险预测")
        with gr.Tabs():
            with gr.Tab("股票价格风险"):
                with gr.Row():
                    with gr.Column():
                        text1 = gr.Textbox(label="公司名称/股票代码", info=""),
                        text2 = gr.Textbox(label="预测日期", info="", value=get_curday),
                        slider = gr.Slider(minimum=1, maximum=4, value=3, step=1, label="历史数据时长")
                        with gr.Row():
                            clear_btn = gr.Button("清除数据")
                            submit_btn = gr.Button("提交数据")
                    with gr.Column():
                        output1 = gr.Textbox(label="提问模板", lines=5)
                        output2 = gr.Textbox(label="预测分析", lines=5)

                clear_btn.click(fn=clean, inputs=[], outputs=[text1, text2, slider, output1, output2])
                submit_btn.click(fn=operate_stock, inputs=[text1, text2, slider], outputs=[output1, output2])
            with gr.Tab("商品价格风险"):
                pass
            with gr.Tab("利率风险"):
                pass
            with gr.Tab("汇率风险"):
                pass

    demo.launch()
