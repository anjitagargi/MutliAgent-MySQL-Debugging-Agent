# DebugSage
### [Uploading DebugSage-multi-agent-mysql.pdf…]()

**A multi-agent assistant that debugs MySQL problems.** Ask a question in plain English. Three specialised agents work in sequence: one reads the right version of the MySQL docs, one checks the real database state with safe read-only queries, and one turns both into specific fixes you can run.

## Demo

<!-- Drag debugsage-demo.mp4 into this spot while editing on GitHub. GitHub uploads it and inserts the video link for you. -->
https://github.com/user-attachments/assets/46f82a3f-88d3-4f83-94fb-c3372a75b954

## Contents

- [The problem](#the-problem)
- [The solution](#the-solution)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Inside the DocSearch Agent](#inside-the-docsearch-agent)
- [Example runs](#example-runs)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Safety notes](#safety-notes)
- [Author](#author)

## The problem

Database administrators and developers lose a lot of time when working with MySQL, especially during version migrations, error troubleshooting and when learning version-specific features. The usual pain points:

- Syntax changes between MySQL versions
- Deprecated keywords, functions and variables
- Modified functionality in newer versions
- Changes in password policies and security features
- Performance optimisation differences
- Varying error messages across versions

The result is slow troubleshooting, potential application failures and delayed upgrades, which keep teams off newer, safer and faster MySQL versions.

## The solution

DebugSage is a multi-agent system that processes each troubleshooting request through three specialised components, one after another.

| # | Agent | What it does |
|---|-------|--------------|
| 1 | **DocSearch Agent** | Analyses MySQL-specific queries, searches versioned documentation (5.7, 8.0, 8.1) and gives version-specific recommendations for version migration, error troubleshooting and general MySQL support. |
| 2 | **DB Analysis Agent** | Runs safe, read-only diagnostic queries to collect the actual database state, guided by what the DocSearch Agent found. |
| 3 | **Solution Agent** | Combines the documentation findings with the live database state into actionable recommendations tailored to your MySQL version and environment. |

The workflow is sequential on purpose: database analysis is guided by the documentation, so the final answer is targeted instead of generic.

## How it works

Before, a DBA or on-call engineer searched documentation, tickets and the database by hand. DebugSage turns that into one guided flow.
1. You ask a question, for example *"Why am I getting replication lag and how can I solve it?"*
2. The **DocSearch Agent** finds the relevant, version-specific documentation.
3. The **DB Analysis Agent** runs read-only diagnostic queries to see what is really happening.
4. The **Solution Agent** returns a final response: a solution overview, steps with SQL commands, verification queries and additional recommendations.

## Architecture
- **Data agents (A1 to A4)** gather the context the model needs: slow queries and max-connection issues, SQL and error logs, documentation trends, and CPU and memory metrics.
- **Vector database** stores that context for retrieval.
- **RAG + LLM** reasons over the retrieved context.
- **SQL agent** looks for checks, validates and executes queries against the database, then feeds the results back to the model before the answer is returned.

## Inside the DocSearch Agent
The DocSearch Agent has two paths so that a failure in one never leaves a question unanswered.

- **Primary path (agent-based reasoning):** the agent picks a tool, such as the troubleshooting handler, and the matching handler builds a prompt from the retrieved documents and the query. The LLM answers with references.
- **Fallback path (prompt-based routing):** if the agent fails, for example when no tool is selected, there is a parsing error or the action is invalid, a prompt router asks the LLM to classify the query as `troubleshooting`, `version_migration` or `documentation_reference`. It then runs the matching chain: `troubleshoot_chain`, `compare_chain` or `doc_ref_chain`.
- **Retrieval:** the query is embedded and the closest document chunks are found with a FAISS search, so every answer comes with its sources.

## Example runs

### Query performance: "How can I improve MySQL query performance?"
The final response included, in order:

1. Setting `innodb_buffer_pool_size` to 85% of the available RAM.
2. Raising `max_connections` for better concurrency.
3. Raising `max_heap_table_size` and `tmp_table_size` to 512 MB.
4. Adding indexes to the `system_logs` and `transactions` tables.
5. Verification commands to confirm each change, plus extra recommendations such as using `EXPLAIN` and proactive monitoring.

### Replication lag: "Why am I getting replication lag and how can I solve it?"

Both runs list steps with the exact commands, how to verify them, and further recommendations such as monitoring replication lag, memory usage and disk I/O.

## Tech stack

- **Agents and chains:** LangChain
- **Retrieval:** FAISS vector store with embeddings, RAG
- **LLM:** a large language model for classification, reasoning and answer generation
- **Knowledge base:** MySQL documentation for versions 5.7, 8.0 and 8.1
- **Database access:** read-only MySQL connection for diagnostics


## Safety notes

- The DB Analysis Agent is meant to run **read-only** diagnostic queries. Connect it with a database user that only has read permissions.
- DebugSage suggests commands, for example `SET GLOBAL ...` or `ALTER TABLE ...`. These are generated by an LLM, so **review every command and test it in a non-production environment first**.
- Recommended values (such as buffer pool and connection limits) depend on your hardware and workload.

## Author

Built by **Anjita**.

- GitHub: [@anjitagargi](https://github.com/anjitagargi)
- LinkedIn: [anjitachandora](https://www.linkedin.com/in/anjitachandora/)
