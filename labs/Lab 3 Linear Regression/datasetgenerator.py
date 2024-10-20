import csv
from datetime import datetime
from collections import defaultdict

# Define the function to compute quarters after Jan 1, 2000
def quarters_after_2000(date_str):
    # Parse the date assuming it's in the format 'DD-MMM-YY'
    date_obj = datetime.strptime(date_str, '%d-%b-%y')
    
    # Define the reference date (Jan 1, 2000)
    reference_date = datetime(2000, 1, 1)
    
    # Calculate the number of quarters between the date and Jan 1, 2000
    years_diff = date_obj.year - reference_date.year
    months_diff = date_obj.month - reference_date.month
    
    total_months = years_diff * 12 + months_diff
    quarters_after = total_months // 3  # Convert months to quarters
    
    return quarters_after

# Dictionary to count occurrences of each quarter after January 1, 2000
quarter_counts = defaultdict(int)

# Read the CSV file
csvfile = open(r'C:\Users\JeanLR\Documents\Projects\DataMining\DATAMINING525HW\labs\Lab 3 Linear Regression\banklist.csv', newline='')
reader = csv.reader(csvfile)

for row in reader:
    # Assuming the date is in the 6th column
    date_str = row[5]
    # Calculate the quarters after Jan 1, 2000
    quarters_after = quarters_after_2000(date_str)
    # Increment the count for that quarter
    quarter_counts[quarters_after] += 1


# Prepare the data for writing
quarters_list = []
counts_list = []

for quarter in sorted(quarter_counts.keys()):
    quarters_list.append(str(quarter))
    counts_list.append(str(quarter_counts[quarter]))


# Write the result to a file called data.txt
outfile = open('data.txt','w')
# Write quarters on the first line
formatted_quarters = ', '.join(f'[{q}]' for q in quarters_list)
outfile.write(formatted_quarters + ',\n')

# Format counts with brackets and write on the second line
formatted_counts = ', '.join(f'[{c}]' for c in counts_list)
outfile.write(formatted_counts + ',\n')