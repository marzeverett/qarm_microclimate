import json
import matplotlib.pyplot as plt 
import pandas as pd


folder_name = "sample_12"
index = 1
#index = "ensemble_or"

key="delta_frost_events"
file_name = f"data/output_data/{folder_name}/{index}_predictions.csv"

df =  pd.read_csv(file_name)



df["time_steps"] = pd.to_datetime(df["time_steps"])

true_df = df[df[key] == 1]
true_df[key] = "true_frost"
pred_df = df[df["predictions"] == 1]
pred_df["predictions"] = "predicted_frost"

title_name = f"Rule Index {index} True vs Predicted"
plt.rcParams["figure.figsize"] = [12.0, 5.0]
plt.rcParams["figure.autolayout"] = True
plt.plot(true_df["time_steps"], true_df[key], 'o', label="true")
plt.plot(pred_df["time_steps"], pred_df["predictions"], 'o', label="predictions")
plt.title(title_name)
plt.legend()
#plt.scatter(df[key], df["predictions"])
#plt.savefig(f"data/output_data/{folder_name}/{index}_compare.png")
plt.show()
plt.clf()