# er-diagram-nlp-verification
#  Automated ER Diagram Verification using NLP and LLMs

##  Overview

This project focuses on automatically verifying whether a **natural language requirement** matches a corresponding **Entity-Relationship (ER) diagram**.

The system evaluates consistency between textual requirements and structured diagrams using multiple approaches, including rule-based methods, semantic embeddings, and Large Language Models (LLMs).

---

##  Problem Statement

In software development, inconsistencies between requirement documents and ER diagrams can lead to incorrect system design.

This project aims to:

* Automatically detect mismatches
* Reduce manual validation effort
* Improve reliability of system design

---

##  Approach

### Dataset Generation

* Source: Kaggle requirements dataset
* Generated 3 dataset types:

  * **Perfect Match**
  * **Synonym Variations**
  * **Logical Errors**

Each dataset contains requirement–diagram pairs labeled as:

* MATCH
* MISMATCH

---

###  Methods Implemented

1. **Basic Rule-Based Matching**

   * Keyword similarity and string matching

2. **Advanced Rule-Based Matching**

   * Synonym mapping + structural checks

3. **Embedding-Based Semantic Matching**

   * Sentence-BERT embeddings + cosine similarity

4. **LLM-Based Reasoning (Main Approach)**

   * Uses LLM to analyze semantics and structure jointly

---

## Results

| Method        | Dataset A | Dataset B | Dataset C |
| ------------- | --------- | --------- | --------- |
| Rule-Based    | 75%       | 10%       | 20%       |
| Advanced Rule | 90%       | 20%       | 5%        |
| Embeddings    | 90%       | 30%       | 15%       |
| **LLM-Based** | **95%**   | **70%**   | **55%**   |

LLM-based method achieved the best performance, especially in detecting logical inconsistencies.

---

##  Technologies Used

* Python
* NLP (Sentence-BERT)
* PlantUML
* OpenAI API (LLM reasoning)
* Data Processing (CSV)

---

##  Project Structure

```bash
src/        # implementation of all methods
data/       # generated datasets
results/    # output CSVs and evaluation
report/     # project report
```

---

## How to Run

1. Install dependencies
2. Run individual method scripts:

```bash
python method1_rule_based.py
python method2_advanced.py
python method3_embeddings.py
python method4_llm.py
```

---

##  Key Insights

* Rule-based methods fail on synonyms
* Embeddings improve semantic understanding
* LLMs provide best performance with logical reasoning

---

##  References

* Kaggle Dataset (Requirements → ER)
* PlantUML
* Sentence-BERT
* OpenAI API

---

##  Author

Nikitha Reddy
