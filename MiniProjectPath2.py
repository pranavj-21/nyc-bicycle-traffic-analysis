import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations


''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total'] = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))


print() # Space

# Question 1: Testing all 3 bridge combinations with regression
print("Q1: Best 3 Bridges")

bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
best_r2 = -1
best_combo = None

for combo in combinations(bridges, 3):
    X = dataset_2[list(combo)]
    y = dataset_2['Total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predicted = model.predict(X_test)
    r2 = r2_score(y_test, predicted)
    
    print(f"Test: {combo}")
    print(f"R^2 = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_combo = combo

print(f"Best combination: {best_combo}")
print(f"R^2 = {best_r2:.4f}")
print()
 

# Question 2
print("Q2: Weather Prediction")

# Creating lag features for the weather data
dataset_2['high_temp_yesterday'] = dataset_2['High Temp'].shift(1)
dataset_2['low_temp_yesterday'] = dataset_2['Low Temp'].shift(1)
dataset_2['rain_yesterday'] = dataset_2['Precipitation'].shift(1)

# Removing the first row   
clean_data = dataset_2.iloc[1:].copy()
X = clean_data[['high_temp_yesterday', 'low_temp_yesterday', 'rain_yesterday']]
y = clean_data['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression model
model_q2 = LinearRegression()
model_q2.fit(X_train, y_train)

predicted_q2 = model_q2.predict(X_test)
r2_q2 = r2_score(y_test, predicted_q2)

print(f"Regression R^2: {r2_q2:.4f}") # R2 value for regression model

print("Model coefficients from equation:")
print(f"High temp: {model_q2.coef_[0]:.2f}")
print(f"Low temp: {model_q2.coef_[1]:.2f}")
print(f"Precipitation: {model_q2.coef_[2]:.2f}")
print(f"Intercept: {model_q2.intercept_:.2f}")
print()
 

# Question 3
print("Q3: Weekly Pattern Analysis")

# Analyzing the weekly pattern
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_traffic = [] # Creating a day_traffic list for graphing purposes

print("Average traffic by day:")
for day in days:
    avg = dataset_2[dataset_2['Day'] == day]['Total'].mean()
    print(f"{day}: {avg:.0f}")
    day_traffic.append(avg)


# Comparing the weekday vs weekend data
weekday_traffic = dataset_2[dataset_2['Day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['Total']
weekend_traffic = dataset_2[dataset_2['Day'].isin(['Saturday', 'Sunday'])]['Total']

weekday_mean = weekday_traffic.mean()
weekend_mean = weekend_traffic.mean()
weekday_std = weekday_traffic.std()
weekend_std = weekend_traffic.std()

print(f"Weekday average: {weekday_mean:.0f} (std: {weekday_std:.0f})")
print(f"Weekend average: {weekend_mean:.0f} (std: {weekend_std:.0f})")
print(f"Difference: {weekday_mean - weekend_mean:.0f}")

# Graph for weekly pattern
# Steel blue for Wednesday (highest), green for other weekdays, and red for weekends
plt.figure(figsize=(10, 6))
plt.bar(days, day_traffic, color=['green', 'green', 'steelblue', 'green', 'green', 'red', 'red'])
plt.xlabel('Day of Week')
plt.ylabel('Average Bike Traffic')
plt.title('Average Daily Bike Traffic by Day of Week')
plt.xticks(rotation=45)
plt.savefig('weekly_pattern.png', dpi=150)
plt.close()