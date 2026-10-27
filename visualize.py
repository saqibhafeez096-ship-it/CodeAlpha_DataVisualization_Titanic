import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")
df = pd.read_csv("titanic_clean.csv")

# Chart 1: Survival by gender
sns.barplot(x="sex", y="survived", data=df, palette="Set2")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.savefig("charts/1_survival_by_gender.png", dpi=150)
plt.close()

# Chart 2: Survival by class
sns.countplot(x="pclass", hue="survived", data=df, palette="Set1")
plt.title("Passenger Class vs Survival")
plt.savefig("charts/2_class_survival.png", dpi=150)
plt.close()

# Chart 3: Age distribution
sns.histplot(df["age"], bins=30, kde=True, color="teal")
plt.title("Age Distribution of Passengers")
plt.savefig("charts/3_age_distribution.png", dpi=150)
plt.close()

# Chart 4: Fare by class
sns.boxplot(x="pclass", y="fare", data=df, palette="pastel")
plt.title("Ticket Fare by Class")
plt.savefig("charts/4_fare_by_class.png", dpi=150)
plt.close()

# Chart 5: Age vs fare scatter, coloured by survival
sns.scatterplot(x="age", y="fare", hue="survived", data=df, alpha=0.7)
plt.title("Age vs Fare (by Survival)")
plt.savefig("charts/5_age_fare_scatter.png", dpi=150)
plt.close()

# Chart 6: Pie chart of embarkation port
df["embark_town"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90)
plt.title("Passengers by Embarkation Port")
plt.ylabel("")
plt.savefig("charts/6_embark_pie.png", dpi=150)
plt.close()

# Chart 7: Dashboard (4 charts in one image)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(x="sex", y="survived", data=df, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Survival by Gender")
sns.countplot(x="pclass", hue="survived", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Class vs Survival")
sns.histplot(df["age"], bins=30, kde=True, ax=axes[1, 0], color="teal")
axes[1, 0].set_title("Age Distribution")
sns.boxplot(x="pclass", y="fare", data=df, ax=axes[1, 1], palette="pastel")
axes[1, 1].set_title("Fare by Class")
fig.suptitle("Titanic Dashboard", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/7_dashboard.png", dpi=150)
plt.close()

print("All charts saved in /charts")