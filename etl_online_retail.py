import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
base = Path(__file__).parent
out = base / "outputs"
out.mkdir(exist_ok=True)

# Load 
df = pd.read_csv(base / "Online Retail.csv", encoding="ISO-8859-1",
                 dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})

# Clean 
df.columns = ["invoice_no", "stock_code", "description", "quantity",
              "invoice_date", "unit_price", "customer_id", "country"]

df["invoice_date"] = pd.to_datetime(df["invoice_date"], dayfirst=True, errors="coerce")
df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce")
df["unit_price"]   = pd.to_numeric(df["unit_price"], errors="coerce")

df = df.dropna(subset=["invoice_no", "stock_code", "invoice_date", "quantity", "unit_price"])
df = df.drop_duplicates()

#  Features 
df["revenue"]       = df["quantity"] * df["unit_price"]
df["is_cancelled"]  = df["invoice_no"].str.upper().str.startswith("C")
df["is_return"]     = df["quantity"] < 0
df["is_valid_sale"] = (~df["is_cancelled"]) & (~df["is_return"]) & \
                      (df["quantity"] > 0) & (df["unit_price"] > 0)

df["year_month"] = df["invoice_date"].dt.strftime("%Y-%m")
df["weekday"]    = df["invoice_date"].dt.day_name()
df["hour"]       = df["invoice_date"].dt.hour

valid = df[df["is_valid_sale"]].copy()
df.to_csv(out / "clean.csv", index=False)

# Chart 1: Monthly Revenue 
monthly = valid.groupby("year_month")["revenue"].sum()

plt.figure(figsize=(12, 5))
monthly.plot(marker="o", color="steelblue")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(out / "chart_monthly.png")
plt.show()

# Chart 2: Top 10 Products 
top_products = valid.groupby("description")["revenue"].sum() \
                    .sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index,
            hue=top_products.index, palette="viridis", legend=False)
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(out / "chart_products.png")
plt.show()

#  Chart 3: Top 10 Countries 
top_country = valid.groupby("country")["revenue"].sum() \
                   .sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_country.values, y=top_country.index,
            hue=top_country.index, palette="magma", legend=False)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(out / "chart_country.png")
plt.show()

#  Chart 4: Revenue by Weekday
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = valid.groupby("weekday")["revenue"].sum().reindex(order)

plt.figure(figsize=(10, 5))
sns.barplot(x=weekday.index, y=weekday.values,
            hue=weekday.index, palette="coolwarm", legend=False)
plt.title("Revenue by Weekday")
plt.tight_layout()
plt.savefig(out / "chart_weekday.png")
plt.show()

#  Chart 5: Revenue by Hour 
hourly = valid.groupby("hour")["revenue"].sum()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="darkorange")
plt.title("Revenue by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(out / "chart_hourly.png")
plt.show()

#  Chart 6: Quantity Distribution (filtered) 
q = valid[valid["quantity"] <= 100]

plt.figure(figsize=(10, 5))
sns.histplot(q["quantity"], bins=30, color="teal")
plt.title("Quantity Distribution (1 to 100)")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(out / "chart_quantity.png")
plt.show()

# Chart 7: Correlation Heatmap 
corr = valid[["quantity", "unit_price", "revenue", "hour"]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(out / "chart_correlation.png")
plt.show()

#  Chart 8: RFM Segments 
rfm = valid[valid["customer_id"].notna()].copy()
snap = rfm["invoice_date"].max() + pd.Timedelta(days=1)

rfm = rfm.groupby("customer_id").agg(
    recency=("invoice_date", lambda x: (snap - x.max()).days),
    frequency=("invoice_no", "nunique"),
    monetary=("revenue", "sum")
)

rfm["r"] = pd.qcut(rfm["recency"].rank(method="first"),   4, labels=[4,3,2,1]).astype(int)
rfm["f"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["m"] = pd.qcut(rfm["monetary"].rank(method="first"),  4, labels=[1,2,3,4]).astype(int)

def label(r, f, m):
    if r == 4 and f == 4 and m == 4: return "Champion"
    if r >= 3 and f >= 3:            return "Loyal"
    if r == 4 and f <= 2:            return "New"
    if r <= 2 and f >= 3:            return "At Risk"
    if r <= 2 and f <= 2:            return "Lost"
    return "Regular"

rfm["segment"] = [label(r, f, m) for r, f, m in zip(rfm["r"], rfm["f"], rfm["m"])]

plt.figure(figsize=(10, 5))
sns.countplot(x="segment", data=rfm,
              hue="segment", palette="Set2", legend=False)
plt.title("Customer Segments (RFM)")
plt.xlabel("")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig(out / "chart_rfm.png")
plt.show()

print("Done. Check the outputs folder.")