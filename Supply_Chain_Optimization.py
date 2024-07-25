import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st

# Load data
order_list = pd.read_csv(r"C:\Users\Selvam\Downloads\OrderList.csv")
freight_rates = pd.read_csv(r"C:\Users\Selvam\Downloads\FreightRates.csv")
plant_ports = pd.read_csv(r"C:\Users\Selvam\Downloads\PlantPorts.csv")
products_per_plant = pd.read_csv(r"C:\Users\Selvam\Downloads\ProductsPerPlant.csv")
vmi_customers = pd.read_csv(r"C:\Users\Selvam\Downloads\VmiCustomers.csv")
wh_capacities = pd.read_csv(r"C:\Users\Selvam\Downloads\WhCapacities.csv")
wh_costs = pd.read_csv(r"C:\Users\Selvam\Downloads\WhCosts.csv")

# Rename columns to match provided names
order_list.columns = ['Order ID', 'Order Date', 'Origin Port', 'Carrier', 'TPT', 'Service Level', 'Ship ahead day count', 'Ship Late Day count', 'Customer', 'Product ID', 'Plant Code', 'Destination Port', 'Unit quantity', 'Weight']
freight_rates.columns = ['Carrier', 'orig_port_cd', 'dest_port_cd', 'minm_wgh_qty', 'max_wgh_qty', 'svc_cd', 'minimum cost', 'rate', 'mode_dsc', 'tpt_day_cnt', 'Carrier type']
plant_ports.columns = ['Plant Code', 'Port']
products_per_plant.columns = ['Plant Code', 'Product ID']
wh_costs.columns = ['Warehouse ID', 'Cost per Unit']
wh_capacities.columns = ['Warehouse ID', 'Daily Capacity']
vmi_customers.columns = ['Warehouse ID', 'Customer']

# Convert necessary columns to numeric
order_list['Unit quantity'] = pd.to_numeric(order_list['Unit quantity'], errors='coerce')
order_list['Weight'] = pd.to_numeric(order_list['Weight'], errors='coerce')
wh_costs['Cost per Unit'] = pd.to_numeric(wh_costs['Cost per Unit'], errors='coerce')

# Merge order list with warehouse costs
order_list = order_list.merge(wh_costs, left_on='Plant Code', right_on='Warehouse ID', how='left')

# Calculate total historical cost (excluding 'Carrier' for now)
order_list['Total_Cost'] = order_list['Unit quantity'] * order_list['Cost per Unit']

# Check for missing values and handle them
order_list['Total_Cost'] = order_list['Total_Cost'].fillna(0)

# Calculate total historical cost
total_historical_cost = order_list['Total_Cost'].sum()

# Analyze capacity utilization
capacity_utilization = order_list.groupby('Plant Code')['Order ID'].count() / wh_capacities.set_index('Warehouse ID')['Daily Capacity']

# Define the cost function: minimize (storage cost + freight cost)
# Assuming you want to minimize the total cost per unit of each product
c = order_list.groupby('Plant Code')['Cost per Unit'].mean().reindex(order_list['Plant Code'].unique()).fillna(0).to_numpy()

# Create A_eq and b_eq for the constraints
order_plant_pivot = pd.pivot_table(order_list, index='Order ID', columns='Plant Code', values='Unit quantity', aggfunc='sum', fill_value=0)
A_eq = order_plant_pivot.to_numpy()
b_eq = order_list.groupby('Order ID')['Unit quantity'].sum().reindex(order_plant_pivot.index).fillna(0).to_numpy()

# Check dimensions
print(f"Dimensions of A_eq: {A_eq.shape}")
print(f"Length of c: {len(c)}")
print(f"Length of b_eq: {len(b_eq)}")

# Ensure the number of columns in A_eq matches the length of c
if A_eq.shape[1] != len(c):
    raise ValueError("Number of columns in A_eq does not match the length of c")

# Optimization
res = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

# Streamlit dashboard
st.title('Supply Chain Optimization Dashboard')

st.subheader('Total Historical Cost')
st.write(total_historical_cost)

st.subheader('Capacity Utilization')
st.bar_chart(capacity_utilization)

st.subheader('Optimal Cost')
st.write(res.fun)

st.subheader('Optimal Routing Solution')
st.write(res.x)
