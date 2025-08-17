import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

# Sample data
dates = [
    datetime.date(2023, 1, 1),
    datetime.date(2023, 1, 2),
    datetime.date(2023, 1, 3),
    datetime.date(2023, 1, 4),
    datetime.date(2023, 1, 5),
]
values = [1, 3, 2, 5, 4]

# Create the plot
fig, ax = plt.subplots()
ax.plot(dates, values)

# Format the x-axis
fig.autofmt_xdate()

# Set labels
ax.set_title('Chart with Calendar X-Axis')
ax.set_xlabel('Date')
ax.set_ylabel('Value')

# Save the figure
plt.savefig('test_chart.png')

print("Chart saved to test_chart.png")
