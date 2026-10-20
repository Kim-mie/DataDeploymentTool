# Exploratory Data Analysis: how to read the plots

Dataset: IBM Telco Customer Churn. 7,043 customers; 7,032 remain after dropping 11 rows with a blank `TotalCharges` (all brand-new customers with tenure 0).
**26.6%** of the remaining customers churned. Every "churn rate" below is compared with that overall rate.

> These plots show **association, not cause**. A feature that lines up with churn is a risk signal, not proof it makes people leave.

---

## 1. Univariate analysis (one variable at a time)

**Target balance (Churn bar chart).** About 73% stayed and 27% left. The classes are imbalanced, so plain accuracy is misleading. A model that always predicts "stay" would score about 73%. This is why the models use class balancing and are judged on recall and ROC-AUC.

**Numeric histograms**

| Feature | What the shape shows |
|---|---|
| `tenure` | **U-shaped.** A tall spike at the very start (about 1,050 customers in the first bar, 1,051 with 3 months or less), a fairly flat middle at roughly 100-300 customers per bar, and a second spike at the maximum of 72 months (about 650 customers). So the base is split between brand-new customers and long-standing ones, with fewer in between. Mean 32, median 29. |
| `MonthlyCharges` | **Three parts.** A huge spike at the cheapest end, about $18-$25 (roughly 1,180 customers in the first bar, 1,402 at $25 or less). Then a dip between about $30 and $45. Then a broad hump from about $65 to $105, peaking around $80. The spike matches the 1,520 customers with **no internet service**, whose bills only range from $18 to $27 (phone only). The hump is the internet customers. Mean $65, median $70, range $18-$119. |
| `TotalCharges` | **Strongly right-skewed.** The tallest bar is at the very lowest values (about 1,500 customers have paid under roughly $300), and the counts fall steadily towards $8,685. A few long-standing, high-paying customers form the long tail. Mean $2,283 is well above the median $1,397. It largely mirrors tenure x monthly charges. |

**Categorical bar charts.** They show how customers are split within each feature. Things to notice:
- Gender is almost 50/50.
- 16% are senior citizens (1,142 customers).
- Most customers have phone service (6,352).
- 1,520 customers have no internet service. That is why many service columns have a third "No internet service" category.
- Month-to-month is the most common contract (3,875), and electronic check is the most common payment method (2,365).

---

## 2. Bivariate analysis (each feature against churn)

**Numeric features by churn (boxplots).**

| | Stayed | Churned |
|---|---|---|
| Mean tenure | 38 months | 18 months |
| Median tenure | 38 months | **10 months** |
| Mean monthly charges | $61 | $74 |

Customers leave **early** and tend to pay **more per month**. Tenure has the strongest numeric link to churn (correlation -0.35). Monthly charges are weaker (+0.19). `TotalCharges` is negatively linked (-0.20) mainly because churners have had less time to accumulate it.

**Churn rate by category (bar charts with the red overall-rate line).** Bars above the line are higher risk.

| Feature | Higher-risk group | Churn rate | Lower-risk group | Churn rate |
|---|---|---|---|---|
| Contract | Month-to-month | **43%** | Two year | **3%** (one year: 11%) |
| Internet service | Fiber optic | **42%** | No internet | 7% (DSL: 19%) |
| Payment method | Electronic check | **45%** | Credit card (auto) | 15% (bank transfer 17%, mailed check 19%) |
| Online security | No | 42% | Yes | 15% |
| Tech support | No | 42% | Yes | 15% |
| Online backup | No | 40% | Yes | 22% |
| Device protection | No | 39% | Yes | 23% |
| Paperless billing | Yes | 34% | No | 16% |
| Senior citizen | Yes | 42% | No | 24% |
| Partner | No | 33% | Yes | 20% |
| Dependents | No | 31% | Yes | 16% |

Features that barely separate churners from stayers: **gender** (27% vs 26%) and **phone service** (25% vs 27%). They carry little signal.

---

## 3. Multivariate analysis (several features together)

**Tenure vs monthly charges, coloured by churn (scatter).** Red points (churned) concentrate at **low tenure and higher monthly charges**. Long-tenure customers, at any price, are mostly blue (stayed).

**Churn % by Contract x Internet service (heatmap).**

| Contract | DSL | Fiber optic | No internet |
|---|---|---|---|
| Month-to-month | 32% | **55%** | 19% |
| One year | 9% | 19% | 2% |
| Two year | 2% | 7% | 1% |

The two effects stack. Fiber optic customers on a month-to-month contract are the riskiest group, with more than half churning. Longer contracts bring churn down sharply in every internet type. So contract length matters more than the internet product itself.

### Correlation matrix (heatmap of the one-hot encoded data)

Red means a positive correlation, blue means a negative one, and white means none. Read the **Churn row** first, then look for blocks of features that move together.

**1. What correlates with churn (the Churn row).**

| Pushes churn UP | r | Pushes churn DOWN | r |
|---|---|---|---|
| Month-to-month contract | **+0.40** | Tenure | **-0.35** |
| No online security | +0.34 | Two-year contract | -0.30 |
| No tech support | +0.34 | No internet service (all the "No internet service" columns) | -0.23 |
| Fiber optic internet | +0.31 | | |
| Electronic check payment | +0.30 | | |
| No online backup | +0.27 | | |
| No device protection | +0.25 | | |
| Monthly charges | +0.19 | | |

None of the values is very large. The strongest single link to churn is about 0.4, so no one feature predicts churn on its own. The models combine many weak signals. Gender and phone service sit close to 0.

**2. Features that move together (the repeated blocks).**

| Pair | r | Meaning |
|---|---|---|
| Tenure and TotalCharges | **+0.83** | Total charges is mostly the length of stay multiplied by the monthly bill. |
| Monthly charges and Fiber optic | +0.79 | Fiber plans are the expensive ones. |
| Monthly charges and TotalCharges | +0.65 | Higher bills build up higher totals. |
| Monthly charges and Streaming TV | +0.63 | Add-on services raise the bill. |
| Tenure and Two-year contract | +0.56 | Long contracts go with long-standing customers. |

**3. Things in the heatmap that are not real findings.**
- **Perfect -1.0 squares.** One-hot encoding makes mirror columns, for example `gender_Female` and `gender_Male`, or `Partner_Yes` and `Partner_No`. They are the same information, so they correlate perfectly. Ignore them.
- **The "No internet service" block.** About six columns (online security, backup, device protection, tech support, streaming TV and movies) share the same 1,520 customers, so they all show the same -0.23 with churn. It is one fact, not six.

**Why it matters for modelling.** Tenure and TotalCharges (0.83) carry a lot of overlapping information, and so do the add-on services. This overlap is why linear model coefficients can be unstable. It's also why the notebook uses permutation importance to rank features, since that handles correlated features better. Tree models and logistic regression with regularisation both cope with it, so keeping all features is reasonable here.

---

## Summary: who is most likely to churn?

A **new** customer on a **month-to-month** contract, on **fiber optic**, paying by **electronic check**, **without** online security or tech support. Customers on one- or two-year contracts with add-on services are the most loyal.

These findings agree with the model's explainability plot, where contract type and tenure are the top drivers.

**Business actions this suggests:** push new customers towards longer contracts, bundle security/support add-ons for fiber customers, and encourage automatic payment methods.
