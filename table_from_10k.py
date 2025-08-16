from sec_api import QueryApi
import pandas as pd
import requests
import io
import json
import sys
import traceback

from dotenv import load_dotenv
import os

def main():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("SEC_API_KEY")
        
        if not api_key:
            print("Error: SEC_API_KEY not found in .env file")
            return 1
        
        # Initialize API client
        queryApi = QueryApi(api_key=api_key)
        
        # Set request headers to simulate browser access
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Query specific company's 10-K filings
        ticker = "AAPL"  # Default ticker
        if len(sys.argv) > 1:
            ticker = sys.argv[1]
            
        print(f"Searching for {ticker} 10-K filings...")
        
        query = {
            "query": { "query_string": { "query": f"ticker:{ticker} AND formType:10-K" }},
            "from": "0",
            "size": "1",
            "sort": [{ "filedAt": "desc" }]
        }
        
        filings = queryApi.get_filings(query)
        
        if not filings or not filings.get('filings') or len(filings['filings']) == 0:
            print(f"No 10-K filings found for {ticker}")
            return 1
        
        # Get the latest 10-K filing details
        filing_url = filings['filings'][0]['linkToFilingDetails']
        accession_no = filings['filings'][0]['accessionNo']
        company_name = filings['filings'][0]['companyNameLong']
        company_cik = filings['filings'][0]['cik']
        filing_date = filings['filings'][0]['filedAt']
        
        print(f"Found filing: {filing_url}")
        print(f"Company CIK: {company_cik}")
        print(f"Filing Date: {filing_date}")
        
        # Extract XBRL data
        xbrl_json = None
        try:
            print("Extracting XBRL data to JSON...")
            xbrl_json = queryApi.xbrl_to_json(filing_url)
            print("Successfully extracted XBRL data")
        except Exception as e:
            print(f"Error extracting XBRL data: {e}")
            
            # Try alternative method to get XBRL data
            try:
                print("Trying alternative method to get XBRL data...")
                # Get filing details
                filing_details = queryApi.get_filing_details(accession_no)
                
                # Find XBRL file URL
                xbrl_url = None
                for document in filing_details.get('documentFormatFiles', []):
                    if document.get('documentType', '').lower() == 'xbrl instance':
                        xbrl_url = document.get('documentUrl')
                        break
                
                if xbrl_url:
                    print(f"Found XBRL URL: {xbrl_url}")
                    # Use requests to get XBRL content directly
                    response = requests.get(xbrl_url, headers=headers)
                    if response.status_code == 200:
                        # Use SEC API to parse XBRL content
                        xbrl_json = queryApi.xbrl_to_json(response.text)
                        print("Successfully extracted XBRL data using alternative method")
                    else:
                        print(f"Failed to download XBRL file. Status code: {response.status_code}")
                else:
                    print("Could not find XBRL file URL")
            except Exception as inner_e:
                print(f"Error with alternative XBRL extraction: {inner_e}")
                traceback.print_exc()
        print("Successfully extracted XBRL data")
    except Exception as e:
        print(f"Error extracting XBRL data: {e}")
    
    # 尝试直接获取XBRL URL
    try:
        print("Trying alternative method to get XBRL data...")
        # 获取filing的详细信息
        filing_details = queryApi.get_filing_details(filings['filings'][0]['accessionNo'])
        
        # 查找XBRL文件URL
        xbrl_url = None
        for document in filing_details.get('documentFormatFiles', []):
            if document.get('documentType', '').lower() == 'xbrl instance':
                xbrl_url = document.get('documentUrl')
                break
        
        if xbrl_url:
            print(f"Found XBRL URL: {xbrl_url}")
            # 使用requests直接获取XBRL内容
            response = requests.get(xbrl_url, headers=headers)
            if response.status_code == 200:
                # 使用SEC API解析XBRL内容
                xbrl_json = queryApi.xbrl_to_json(response.text)
                print("Successfully extracted XBRL data using alternative method")
            else:
                print(f"Failed to download XBRL file. Status code: {response.status_code}")
                xbrl_json = None
        else:
            print("Could not find XBRL file URL")
            xbrl_json = None
    except Exception as inner_e:
        print(f"Error with alternative XBRL extraction: {inner_e}")
        traceback.print_exc()
        xbrl_json = None

# 从 XBRL中抽取财务报表
if xbrl_json:
    try:
        print("Extracting balance sheet...")
        balance_sheet = queryApi.get_balance_sheet(xbrl_json)
        print("Successfully extracted balance sheet")
        
        print("Extracting income statement...")
        income_statement = queryApi.get_income_statement(xbrl_json)
        print("Successfully extracted income statement")
        
        print("Extracting cash flow statement...")
        cash_flow = queryApi.get_cash_flow_statement(xbrl_json)
        print("Successfully extracted cash flow statement")
        
        # 转成DataFrame，方便后续分析
        df_bs = pd.DataFrame(balance_sheet)
        df_is = pd.DataFrame(income_statement)
        df_cf = pd.DataFrame(cash_flow)
        
        # 导出为Excel做查看
        with pd.ExcelWriter('Financial_Statements_AAPL.xlsx') as writer:
            df_bs.to_excel(writer, sheet_name='Balance Sheet')
            df_is.to_excel(writer, sheet_name='Income Statement')
            df_cf.to_excel(writer, sheet_name='Cash Flow Statement')
        
        print("资产负债表、利润表和现金流量表已导出至Excel")
    except Exception as e:
        print(f"Error extracting financial statements from XBRL: {e}")
        traceback.print_exc()
        print("Falling back to HTML table extraction...")
        balance_sheet = None
        income_statement = None
        cash_flow = None
else:
    print("No XBRL data available, falling back to HTML table extraction...")
    balance_sheet = None
    income_statement = None
    cash_flow = None

# 获取最新10-K的详细信息
filing_url = filings['filings'][0]['linkToFilingDetails']
accession_no = filings['filings'][0]['accessionNo']
company_name = filings['filings'][0]['companyNameLong']
company_cik = filings['filings'][0]['cik']
filing_date = filings['filings'][0]['filedAt']

print(f"Company CIK: {company_cik}")
print(f"Filing Date: {filing_date}")
df_cf.to_excel(writer, sheet_name='Cash Flow Statement')

print("资产负债表、利润表和现金流量表已导出至Excel")


# 获取最新10-K的详细信息
filing_url = filings['filings'][0]['linkToFilingDetails']
accession_no = filings['filings'][0]['accessionNo']
cik = filings['filings'][0]['cik']
form_type = filings['filings'][0]['formType']
filing_date = filings['filings'][0]['filedAt']

print(f"Found filing: {filing_url}")
print(f"Company CIK: {cik}")
print(f"Filing Date: {filing_date}")

try:
    # 检查API密钥是否存在
    if not api_key:
        print("Error: SEC_API_KEY not found in environment variables.")
        sys.exit(1)
        
    # 检查查询结果是否有效
    if not filings or 'filings' not in filings or not filings['filings']:
        print("Error: No filings found for the query.")
        print(f"Query response: {json.dumps(filings, indent=2)}")
        sys.exit(1)
        
    # 直接从SEC Edgar网站获取HTML内容
    print("Downloading filing HTML content...")
    response = requests.get(filing_url)
    
    # 检查响应状态
    if response.status_code != 200:
        print(f"Error: Failed to download filing. Status code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        sys.exit(1)
        
    html_content = response.text
    print(f"Downloaded {len(html_content)} bytes of HTML content")
    
    # 保存HTML内容到文件以便调试
    with open('sec_filing.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Saved HTML content to sec_filing.html")
    
    # 使用pandas读取HTML中的表格
    print("Extracting tables from HTML...")
    try:
        tables = pd.read_html(html_content)
        print(f"Found {len(tables)} tables in the filing")
    except Exception as e:
        print(f"Error extracting tables: {e}")
        print("Trying with a different parser...")
        try:
            # 尝试使用不同的解析器
            tables = pd.read_html(html_content, flavor='bs4')
            print(f"Found {len(tables)} tables using bs4 parser")
        except Exception as e2:
            print(f"Failed with bs4 parser too: {e2}")
            # 保存一个小样本以便调试
            with open('sample.html', 'w', encoding='utf-8') as f:
                f.write(html_content[:10000])
            print("Saved a sample of the HTML to sample.html")
            sys.exit(1)
    
    # 创建一个字典来存储识别出的财务表格
    financial_tables = {}
    
    # 尝试识别资产负债表、利润表和现金流量表
    for i, table in enumerate(tables):
        try:
            # 检查表格的列名和内容，尝试确定表格类型
            table_str = str(table).lower()
            
            if any(keyword in table_str for keyword in ['balance sheet', 'assets', 'liabilities']):
                financial_tables['balance_sheet'] = table
                print(f"Table {i}: Identified as Balance Sheet")
                
            elif any(keyword in table_str for keyword in ['income statement', 'revenue', 'earnings']):
                financial_tables['income_statement'] = table
                print(f"Table {i}: Identified as Income Statement")
                
            elif any(keyword in table_str for keyword in ['cash flow', 'operating activities']):
                financial_tables['cash_flow'] = table
                print(f"Table {i}: Identified as Cash Flow Statement")
        except Exception as table_e:
            print(f"Error processing table {i}: {table_e}")
    
    print(f"Identified {len(financial_tables)} financial tables")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)

# 导出识别出的财务表格到Excel
if financial_tables:
    print("Exporting identified financial tables to Excel...")
    with pd.ExcelWriter('Financial_Statements_AAPL.xlsx') as writer:
        # 导出每个识别出的财务表格
        for table_name, table_df in financial_tables.items():
            table_df.to_excel(writer, sheet_name=table_name.replace('_', ' ').title())
        
        # 导出前10个表格供参考
        for i, table in enumerate(tables[:10]):
            if i < 10:  # 只导出前10个表格
                sheet_name = f"Table_{i}"
                table.to_excel(writer, sheet_name=sheet_name)
    
    print("财务表格已导出至 Financial_Statements_AAPL.xlsx")
else:
    print("未能识别出财务表格，将导出所有表格供参考")
    
    # 如果没有识别出财务表格，就导出所有表格
    with pd.ExcelWriter('All_Tables_AAPL.xlsx') as writer:
        for i, table in enumerate(tables):
            if i < 30:  # 限制导出表格数量
                sheet_name = f"Table_{i}"
                table.to_excel(writer, sheet_name=sheet_name)
    
    print("所有表格已导出至 All_Tables_AAPL.xlsx")
