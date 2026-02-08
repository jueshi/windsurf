from collections import Counter
from pathlib import Path

white = Counter()
red = Counter()

data_path = Path(__file__).with_name("Lottery_Powerball_Winning_Numbers__Beginning_2010.csv")
with data_path.open(encoding="utf-8") as f:
    next(f)  # 跳过表头
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        nums = parts[1].split()
        if len(nums) != 6:
            continue
        whites = list(map(int, nums[:5]))
        r = int(nums[5])
        white.update(whites)
        red.update([r])

print("白球出现次数最多的前 10 个：", white.most_common(10))
print("红球出现次数最多的前 10 个：", red.most_common(10))
