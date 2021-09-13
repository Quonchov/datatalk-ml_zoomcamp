# Import Functions
import pandas as pd
import numpy as np

# Question 1
# What's the version of NumPy that you installed? You can get the version information using the __version__ field: np.__version__
" Ans = 1.19.5 "

# Question 2
# What's the version of Pandas?
" Ans = 1.3.2 "

# Reading csv files with pandas
car_data = pd.read_csv(y) # y is the directory for the data
# print(car_data)

# Question 3: Average price of BMW cars
mean_car_price = car_data.groupby("Make").MSRP.mean() # Groups the cars make by the average car price
bmw_av_price = mean_car_price["BMW"]  # Gets the average BMW car price
print(f"The average price of BMW: {bmw_av_price}")

# Question 4
car_year = car_data[(car_data["Year"] >= 2015)] # Shows a list the of car data from 2015 to recent
missing_val = car_year.isnull().sum()   # Sums the missing values for each car feature
hp_mis_val = car_year["Engine HP"].isnull().sum()
print(f"The number of missing values in Engine HP: {hp_mis_val}") # print the number of missing values for Engine HP

# Question 5
mean_hp_before = car_data["Engine HP"].mean()  # mean Engine HP
mean_hp_before = round(mean_hp_before)
print(f"Mean HP before: {mean_hp_before}")
car_data["Engine HP"] = car_data["Engine HP"].fillna(value=mean_hp_before)
mean_hp_after = car_data["Engine HP"].mean()
mean_hp_after = round(mean_hp_after)
print(f"Mean HP after: {mean_hp_after}")

# Question 6
car_rr = car_data[car_data.Make == "Rolls-Royce"] # Rolls-Royce information
rr_details = car_rr[["Make","Engine HP","Engine Cylinders","highway MPG"]] # Rolls-Royce specific information 
rr_details = rr_details.drop_duplicates(subset=["Engine HP","Engine Cylinders","highway MPG"]) # Drops the duplicate values in the data
X = np.array([rr_details["Engine HP"],rr_details["Engine Cylinders"],rr_details["highway MPG"]])
XT = X.transpose() # Transpose of X
XTX = X.dot(XT)  # matrix-matrix multiplication
XTX_inv = np.linalg.inv(XTX)  # inverse of the matrix
sum_XTX = XTX_inv.sum()
print(f"The sum of elements of the inverse matrix: {sum_XTX}")


# Questions 7
y = np.array([1000,1100,900,1200,1000,850,1300])  # Array of y values 
mul_val = XT.dot(XTX_inv)  # Multiplication of the inverse of XTX with the transpose of X
w = y.dot(mul_val) # Normal Equation result
print(f"The first value of w: {w[0]}")
