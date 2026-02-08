import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Paste the raw text data here
data = """
Insider TradingRelationshipDateTransactionCost#SharesValue ($)#Shares TotalSEC Form 4ZHAN (BVI) CO LTD.DirectorDec 11 '25Proposed Sale157.98370,00058,452,600Dec 11 04:26 PMCheng Chi FungChief Technology OfficerDec 08 '25Sale174.7055,0009,608,2356,613,961Dec 10 04:08 PMKhaira ManpreetDirectorDec 08 '25Sale175.002,000350,00058,114Dec 10 04:08 PMACEVEDO SYLVIADirectorDec 05 '25Sale185.171,875347,19421,098Dec 09 06:23 PMMANPREET KHAIRADirectorDec 08 '25Proposed Sale175.002,000350,000Dec 08 04:59 PMSYLVIA ACEVEDODirectorDec 05 '25Proposed Sale185.171,875347,194Dec 05 04:45 PMSutardja, PantasDirectorNov 03 '25Sale189.351,875355,0315,893,909Nov 05 04:50 PMBrennan William JosephPres & Chief Executive OfficerOct 30 '25Sale171.2350,0008,561,4381,942,502Nov 03 04:31 PMBrennan William JosephPres & Chief Executive OfficerOct 30 '25Sale171.2318,0163,084,960319,907Nov 03 04:31 PMPANTAS SUTARDJADirectorNov 03 '25Proposed Sale187.623,750703,575Nov 03 04:24 PMLam Yat TungChief Operating OfficerOct 29 '25Sale169.2780,00013,541,968770,000Oct 31 07:45 PMCheng Chi FungChief Technology OfficerOct 27 '25Sale156.3255,0008,597,4186,668,961Oct 29 04:30 PMLam Yat TungChief Operating OfficerOct 22 '25Sale137.4970,0009,624,043850,000Oct 24 04:48 PMCheng Chi FungChief Technology OfficerOct 20 '25Sale150.3455,0008,268,8896,723,961Oct 22 04:11 PMEVELYN JOB & APRIL FOUNDATIONDirectorOct 15 '25Proposed Sale129.7530,0003,892,500Oct 15 04:29 PMCheng Chi FungChief Technology OfficerOct 13 '25Sale148.0855,0008,144,6126,778,961Oct 15 04:04 PMSutardja, PantasDirectorOct 13 '25Sale147.414,408649,7615,894,244Oct 15 04:04 PMKhaira ManpreetDirectorOct 13 '25Sale149.675,000748,36258,574Oct 15 04:03 PMKhaira ManpreetDirectorOct 13 '25Proposed Sale149.905,000749,500Oct 14 01:55 PMPANTAS SUTARDJADirectorOct 13 '25Proposed Sale147.414,408649,761Oct 14 08:53 AMCHENG HUANG FAMILY TRUSTDirectorOct 13 '25Proposed Sale138.83385,00053,449,550Oct 14 07:41 AMFleming Daniel W.Chief Financial OfficerOct 08 '25Sale138.20112,58015,558,280458,678Oct 10 04:49 PMBrennan William JosephPres & Chief Executive OfficerOct 08 '25Sale143.129,4761,356,211337,923Oct 09 04:38 PMBrennan William JosephPres & Chief Executive OfficerOct 07 '25Sale140.069,4761,327,207347,399Oct 09 04:38 PMDaniel FlemingOfficerOct 08 '25Proposed Sale148.87112,58016,759,785Oct 08 04:55 PMWilliam Joseph BrennanOfficerOct 07 '25Proposed Sale137.20154,98421,263,805Oct 07 04:43 PMCheng Chi FungChief Technology OfficerOct 06 '25Sale148.3555,0008,159,2036,833,961Oct 07 04:04 PMLam Yat TungChief Operating OfficerOct 01 '25Sale143.5380,00011,482,770920,000Oct 03 04:54 PMCheng Chi FungChief Technology OfficerSep 29 '25Sale148.8755,0008,187,6866,888,961Oct 01 04:32 PMZHAN (BVI) CO LTD.DirectorOct 01 '25Proposed Sale145.61230,00033,490,300Oct 01 04:32 PMCheng Chi FungChief Technology OfficerSep 22 '25Sale164.9855,0009,074,1236,943,961Sep 24 04:05 PMTAN LIP BUDirectorSep 18 '25Sale175.1220,0003,502,478422,180Sep 22 04:47 PMBrennan William JosephPres & Chief Executive OfficerSep 16 '25Sale164.1150,0008,205,5141,992,502Sep 18 05:41 PMBrennan William JosephPres & Chief Executive OfficerSep 16 '25Sale164.1118,0162,956,612369,173Sep 18 05:41 PMWALDEN TECHNOLOGY VENTURES IIDirectorSep 18 '25Proposed Sale175.1220,0003,502,478Sep 18 04:25 PMCheng Chi FungChief Technology OfficerSep 15 '25Sale163.9855,0009,018,7986,998,961Sep 17 04:15 PMTAN LIP BUDirectorSep 12 '25Sale161.9825,0004,049,457442,180Sep 16 04:04 PMBrennan William JosephPres & Chief Executive OfficerSep 11 '25Sale161.269,4761,528,1322,052,502Sep 15 04:05 PMWALDEN TECHNOLOGY VENTURES IIDirectorSep 12 '25Proposed Sale161.9725,0004,049,250Sep 12 04:54 PMWilliam Joseph BrennanOfficerSep 11 '25Proposed Sale159.3277,49212,346,025Sep 11 05:36 PMCheng Chi FungChief Technology OfficerSep 08 '25Sale146.2155,0008,041,7537,053,961Sep 10 04:46 PMFleming Daniel W.Chief Financial OfficerSep 05 '25Sale137.553,790521,326576,178Sep 09 04:15 PMLaufman JamesChief Legal Officer, SecretarySep 05 '25Sale140.0010,0001,400,019232,284Sep 09 04:15 PMDaniel FlemingOfficerSep 05 '25Proposed Sale140.823,790533,708Sep 05 04:43 PMJAMES L LAUFMANOfficerSep 05 '25Proposed Sale140.0010,0001,400,019Sep 05 04:04 PMBrennan William JosephPres & Chief Executive OfficerAug 01 '25Sale106.87150,00016,031,1362,061,978Aug 05 05:05 PMBrennan William JosephPres & Chief Executive OfficerAug 01 '25Sale106.8746,4444,963,664393,338Aug 05 05:05 PMSutardja, PantasDirectorAug 01 '25Sale107.001,875200,6245,898,652Aug 05 05:05 PMCheng Chi FungChief Technology OfficerAug 01 '25Sale107.1255,0005,891,6577,108,961Aug 05 05:05 PMWilliam Joseph BrennanOfficerAug 01 '25Proposed Sale107.56196,44421,129,517Aug 01 04:41 PMPANTAS SUTARDJADirectorAug 01 '25Proposed Sale111.551,875209,156Aug 01 04:29 PMCheng Chi FungChief Technology OfficerJul 28 '25Sale105.8055,0005,819,1587,163,961Jul 30 04:44 PMCheng Chi FungChief Technology OfficerJul 21 '25Sale96.5455,0005,309,5307,218,961Jul 23 05:29 PMCheng Chi FungChief Technology OfficerJul 14 '25Sale98.5355,0005,419,1447,273,961Jul 16 06:10 PMFleming Daniel W.Chief Financial OfficerJul 10 '25Sale97.293,790368,733582,428Jul 14 07:34 PMCheng Chi FungChief Technology OfficerJul 07 '25Sale91.9455,0005,056,6977,328,961Jul 08 05:11 PMCHENG HUANG FAMILY TRUSTDirectorJul 07 '25Proposed Sale93.61550,00051,485,500Jul 07 04:15 PMCheng Chi FungChief Technology OfficerJun 30 '25Sale92.6555,0005,095,5827,383,961Jul 02 04:23 PMTAN LIP BUDirectorJun 24 '25Sale90.7780,0007,261,816479,428Jun 26 05:02 PMTAN LIP BUDirectorJun 25 '25Sale93.2212,2481,141,729467,180Jun 26 05:02 PMCheng Chi FungChief Technology OfficerJun 23 '25Sale84.3755,0004,640,1777,438,961Jun 25 04:42 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 25 '25Proposed Sale93.2212,2481,141,729Jun 25 04:08 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 24 '25Proposed Sale90.7780,0007,261,816Jun 24 04:28 PMLaufman JamesChief Legal Officer, SecretaryJun 20 '25Sale85.075,000425,350249,346Jun 20 05:43 PMTAN LIP BUDirectorJun 18 '25Sale86.3854,2974,690,305559,428Jun 20 05:39 PMJAMES L LAUFMANOfficerJun 20 '25Proposed Sale85.075,000425,350Jun 20 04:30 PMTAN LIP BUDirectorJun 16 '25Sale77.90171,47313,358,179613,725Jun 18 04:47 PMCheng Chi FungChief Technology OfficerJun 16 '25Sale78.7755,0004,332,3507,493,961Jun 18 04:25 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 18 '25Proposed Sale86.3854,2974,690,305Jun 18 04:22 PMHOSEIN CLYDEDirectorJun 17 '25Sale78.492,000156,97418,761Jun 18 04:19 PMCLYDE R. HOSEINDirectorJun 17 '25Proposed Sale78.492,000156,974Jun 17 04:36 PMTAN LIP BUDirectorJun 12 '25Sale75.33100,0007,533,380788,725Jun 16 04:25 PMTAN LIP BUDirectorJun 13 '25Sale76.053,527268,221785,198Jun 16 04:25 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 16 '25Proposed Sale77.90171,47313,358,175Jun 16 04:13 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 13 '25Proposed Sale76.053,527268,221Jun 13 04:17 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 12 '25Proposed Sale75.33100,0007,533,380Jun 12 04:37 PMLaufman JamesChief Legal Officer, SecretaryJun 11 '25Sale71.5410,000715,352254,346Jun 12 04:36 PMJAMES L LAUFMANOfficerJun 11 '25Proposed Sale71.5410,000715,352Jun 11 05:26 PMCheng Chi FungChief Technology OfficerJun 05 '25Sale74.9659,6414,470,4277,603,961Jun 09 06:56 PMCheng Chi FungChief Technology OfficerJun 09 '25Sale72.5355,0003,989,3297,548,961Jun 09 06:56 PMTAN LIP BUDirectorJun 05 '25Sale75.2075,0005,640,261888,725Jun 09 06:26 PMFleming Daniel W.Chief Financial OfficerJun 06 '25Sale73.5712,498919,456589,854Jun 09 06:25 PMFleming Daniel W.Chief Financial OfficerJun 05 '25Sale77.353,790293,153602,352Jun 09 06:25 PMDaniel FlemingOfficerJun 05 '25Proposed Sale73.0020,0781,465,694Jun 05 05:42 PMCHENG HUANG FAMILY TRUSTOfficerJun 05 '25Proposed Sale74.9659,6414,470,427Jun 05 04:36 PMWALDEN TECHNOLOGY VENTURES IIDirectorJun 05 '25Proposed Sale75.2075,0005,640,255Jun 05 04:20 PMZinsner DavidDirectorJun 04 '25Proposed Sale73.107,500548,250Jun 04 01:08 PMCheng Chi FungChief Technology OfficerMay 02 '25Sale48.0255,0002,640,9657,663,602May 06 05:49 PMDZHS COMMUNITY PROPERTY TRUSTFormer DirectorMay 05 '25Proposed Sale49.0213,333653,633May 05 04:06 PMSutardja, PantasDirectorMay 01 '25Sale46.011,87586,2695,900,527May 05 03:12 PMPANTAS SUTARDJADirectorMay 01 '25Proposed Sale43.051,87580,719May 01 04:22 PMCheng Chi FungChief Technology OfficerApr 28 '25Sale42.9555,0002,362,2817,718,602Apr 30 04:31 PMCheng Chi FungChief Technology OfficerApr 22 '25Sale36.7455,0002,020,6327,773,602Apr 24 04:55 PMCheng Chi FungChief Technology OfficerApr 16 '25Sale37.2955,0002,051,0237,828,602Apr 18 04:24 PMLaufman JamesChief Legal Officer, SecretaryApr 14 '25Sale39.098,000312,720271,725Apr 16 04:30 PMFleming Daniel W.Chief Financial OfficerApr 11 '25Sale40.753,790154,453608,602Apr 15 06:30 PMCheng Chi FungChief Technology OfficerApr 10 '25Sale38.7855,0002,133,0287,883,602Apr 14 05:12 PMLaufman JamesOfficerApr 14 '25Proposed Sale40.008,000320,000Apr 14 04:31 PMDaniel FlemingOfficerApr 10 '25Proposed Sale38.243,790144,930Apr 10 04:53 PMCheng Chi FungChief Technology OfficerApr 04 '25Sale32.9655,0001,812,8147,938,602Apr 08 04:16 PM
"""

ENTRY_PATTERN = re.compile(
    r"""
    (?P<who>.+?)                               # Insider name + title blob
    (?P<date>[A-Z][a-z]{2}\s\d{1,2}\s'\d{2})     # e.g., Dec 11 '25 or Dec 5 '25
    (?P<trans>Proposed\sSale|Sale)             # Transaction type
    (?P<cost>\d+(?:\.\d+)?)                    # Cost / share
    (?P<shares>[\d,]+)                         # Number of shares
    (?P<value>[\d,]+)                          # Dollar value
    (?P<timestamp>[A-Z][a-z]{2}\s\d{1,2}\s\d{2}:\d{2}\s[AP]M)  # Dec 11 04:26 PM or Dec 5 04:26 PM
    """,
    re.VERBOSE | re.DOTALL
)


def _clean_raw_text(raw_text: str) -> str:
    """Normalize whitespace and drop the Finviz header blurb."""
    if not raw_text:
        return ""
    cleaned = raw_text.replace('\r', '').replace('\xa0', ' ')
    if "SEC Form 4" in cleaned:
        cleaned = cleaned.split("SEC Form 4", 1)[1]
    return cleaned.strip()


def parse_transactions(raw_text: str):
    """
    Parse Finviz insider tables whether they are pasted as one long line or
    already newline-separated. Falls back to line parsing if needed.
    """
    cleaned = _clean_raw_text(raw_text)
    transactions = []

    for match in ENTRY_PATTERN.finditer(cleaned):
        transactions.append({
            'Who': match.group('who').strip(),
            'Date': match.group('date'),
            'Transaction': match.group('trans').replace('  ', ' ').strip(),
            'Cost': float(match.group('cost')),
            'Shares': int(match.group('shares').replace(',', '')),
            'Value': int(match.group('value').replace(',', ''))
        })

    if transactions:
        return transactions

    # Fallback: attempt line-by-line parsing similar to the original approach.
    normalized = re.sub(r'(?<=[AP]M)(?=[A-Z])', '\n', cleaned)
    for line in (ln.strip() for ln in normalized.splitlines() if ln.strip()):
        line = re.sub(r"[A-Z][a-z]{2} \d{1,2} \d{2}:\d{2} [AP]M$", "", line).strip()

        date_match = re.search(r'([A-Z][a-z]{2} \d{1,2} \'\d{2})', line)
        if not date_match:
            continue
        date_str = date_match.group(1)
        idx = line.find(date_str)
        who_raw = line[:idx]
        post_date = line[idx + len(date_str):]
        trans_match = re.match(r'(Proposed Sale|Sale)([\d\.]+)([\d,]+)([\d,]+)', post_date)
        if not trans_match:
            continue
        transactions.append({
            'Who': who_raw,
            'Date': date_str,
            'Transaction': trans_match.group(1),
            'Cost': float(trans_match.group(2)),
            'Shares': int(trans_match.group(3).replace(',', '')),
            'Value': int(trans_match.group(4).replace(',', ''))
        })

    return transactions


# --- Data Processing ---

parsed_data = parse_transactions(data)

df = pd.DataFrame(parsed_data)
df['Date'] = pd.to_datetime(df['Date'], format='%b %d \'%y')

# Filter for only 'Sale' transactions (exclude 'Proposed Sale')
sales_df = df[df['Transaction'] == 'Sale'].copy()
sales_df = sales_df.sort_values('Date')

# Function to clean up long job titles from names
def clean_name(s):
    titles = [
        "Pres & Chief Executive Officer", "Chief Technology Officer", 
        "Chief Operating Officer", "Chief Financial Officer",
        "Chief Legal Officer, Secretary", "Director", "Officer", 
        "Former Director"
    ]
    # Custom mapping for clarity
    if "ZHAN (BVI)" in s: return "ZHAN (BVI)"
    if "WALDEN" in s: return "WALDEN TECH"
    if "Brennan" in s: return "Brennan William"
    if "Cheng Chi" in s: return "Cheng Chi Fung"
    if "TAN LIP" in s: return "Lip-Bu Tan"
    
    # Strip standard titles
    for t in titles:
        if s.endswith(t):
            return s[:-len(t)]
    return s

sales_df['Name'] = sales_df['Who'].apply(clean_name)

# Group names: Keep top 6 most active, label rest as "Other"
top_names = sales_df['Name'].value_counts().index[:6]
sales_df['NameGroup'] = sales_df['Name'].apply(lambda x: x if x in top_names else 'Other')

# Create a proxy for Stock Price (Average transaction cost per day)
price_trend = sales_df.groupby('Date')['Cost'].mean().reset_index()

# --- Plotting ---

plt.figure(figsize=(14, 7))

# 1. Plot the Stock Price Line
plt.plot(price_trend['Date'], price_trend['Cost'], 
         color='gray', alpha=0.4, linestyle='--', linewidth=1, label='Avg Transaction Price')

# 2. Plot the Bubble Chart
sns.scatterplot(
    data=sales_df, 
    x='Date', 
    y='Cost', 
    size='Value',       # Bubble size = Transaction Value
    hue='NameGroup',    # Color = Person
    sizes=(50, 1000),   # Range of bubble sizes
    alpha=0.8, 
    edgecolor='black',
    palette='deep'
)

# 3. Annotate the largest transactions
top_transactions = sales_df.nlargest(4, 'Value')
for idx, row in top_transactions.iterrows():
    label = f"${row['Value']/1e6:.1f}M"
    plt.annotate(
        label, 
        (row['Date'], row['Cost']),
        xytext=(0, 10), 
        textcoords='offset points',
        ha='center', 
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

# Formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.ylabel('Share Price ($)')
plt.xlabel('Date (2025)')
plt.title('Insider Sales Transactions: Credo Technology Group (CRDO)', fontsize=14)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Insider')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()