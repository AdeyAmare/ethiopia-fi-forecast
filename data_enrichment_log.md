# Data Enrichment Log

This section documents all additional records introduced during Task 1, explaining **what was added**, **why it matters**, and **how it supports forecasting financial inclusion in Ethiopia**.

---

## **REC_0034 — Smartphone Penetration Rate (Observation)**

**What was added**
An observation capturing Ethiopia’s estimated **smartphone penetration rate** for 2024.

**Why this was added**
Smartphone ownership is a **critical prerequisite** for most digital financial services, including mobile money applications, QR payments, and app-based banking. While the Global Findex measures *outcomes* such as account ownership and digital payment usage, it does not directly measure whether people have the **necessary devices** to use these services.

**Reasoning and relevance**

* Smartphone penetration acts as a **leading indicator** for future growth in digital payments.
* A low smartphone penetration rate helps explain why digital payment usage may lag behind the rapid growth in registered mobile money accounts.
* Including this variable allows the forecasting model to distinguish between **supply-side expansion** (more services available) and **user readiness** (ability to access and use them).

**Why it improves forecasting**
By incorporating smartphone penetration, the model can better estimate **upper bounds** on digital payment adoption and avoid overestimating usage growth in the short term.

---

## **REC_0035 — Internet Usage Rate (Observation)**

**What was added**
An observation representing the percentage of adults using the internet in Ethiopia (2024).

**Why this was added**
Digital payments increasingly depend on **mobile data connectivity**, especially for app-based transactions, merchant payments, and online services. Internet usage is therefore a key **enabling condition** for digital financial inclusion.

**Reasoning and relevance**

* Even if individuals own accounts and smartphones, low internet usage can limit **active usage** of digital financial services.
* This indicator helps explain discrepancies between:

  * High numbers of registered accounts
  * Lower levels of reported digital payment usage in Findex surveys

**Why it improves forecasting**
Including internet usage allows the model to:

* Better estimate **usage growth trajectories**
* Identify structural constraints that slow adoption
* Produce more realistic medium-term forecasts for 2025–2027

---

## **EVT_NEW_003 — Agent Banking Regulation Issued (Event)**

**What was added**
A regulatory event marking the issuance of **agent banking regulations** by the National Bank of Ethiopia.

**Why this was added**
Agent networks are a **foundational access channel** in Ethiopia, particularly for rural, low-income, and first-time users. This regulatory change enabled banks and mobile money providers to expand physical access points beyond traditional branches.

**Reasoning and relevance**

* Agent banking reduces distance, cost, and documentation barriers.
* It is especially important for explaining **access gains outside urban centers**.
* The event was not explicitly captured in the original dataset, despite its clear relevance to financial inclusion outcomes.

**Why it improves forecasting**
Including this event allows the model to:

* Attribute part of access growth to regulatory change rather than pure market dynamics
* Better explain regional and rural inclusion trends
* Model delayed but sustained impacts on account ownership

---

## **LNK_0004 — Fayda Digital ID → Account Ownership (Impact Link)**

**What was added**
An impact link connecting the **Fayda Digital ID rollout** to **Account Ownership**, with an enabling relationship and a delayed effect.

**Why this link was added**
Lack of formal identification is a major barrier to financial inclusion. Digital ID systems reduce Know-Your-Customer (KYC) costs and simplify onboarding for both banks and mobile money providers.

**Reasoning and relevance**

* Fayda does not immediately increase account ownership; its impact materializes **over time** as institutions integrate it into onboarding processes.
* Evidence from other countries shows strong links between digital ID programs and inclusion gains.

**Why it improves forecasting**
This link allows the forecasting model to:

* Represent **time-lagged policy impacts**
* Avoid incorrectly assuming immediate effects
* Incorporate realistic, medium-term access acceleration scenarios

---

## **LNK_0005 — Telebirr Launch → Digital Payment Usage (Impact Link)**

**What was added**
An impact link connecting the **Telebirr launch** to **Digital Payment Usage**, modeled as a direct and high-impact relationship.

**Why this link was added**
Telebirr is not merely an account-opening product; it is a **high-frequency transaction platform** widely used for P2P transfers, merchant payments, and bill payments.

**Reasoning and relevance**

* Observed transaction data shows sharp increases in P2P volumes following Telebirr’s launch.
* The strongest effect is on **usage**, not just access.
* The impact occurs relatively quickly, as users can transact immediately after onboarding.

**Why it improves forecasting**
This link enables the model to:

* Correctly assign large usage gains to product innovation
* Distinguish between account ownership growth and actual financial activity
* Validate modeled impacts against observed transaction data

---

## **Overall Value of the Enrichment**

Together, these additions shift the dataset from being **purely descriptive** to being **explanatory and predictive**. They allow the forecasting system to answer not only *what happened*, but *why it happened* and *what is likely to happen next*.

---