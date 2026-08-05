# AgentLnc

AgentLnc is a local, multi-skill research agent for lncRNA-RBP analysis. It provides three command-line skills and a PyQt5 desktop GUI.

## Requirements and installation (read this first)

No `environment.yml` file is distributed with this project. Users must install the runtime packages themselves.

AgentLnc was validated with `/opt/anaconda3/envs/agentlnc/bin/python` on macOS ARM64. The verified Python version is **Python 3.10.18**.

### Required Python packages

| Package | Verified version | Used for |
|---|---:|---|
| `numpy` | 1.26.4 | Numeric arrays and FAISS data |
| `pandas` | 2.3.2 | Local tables and report data |
| `faiss-cpu` (`import faiss`) | 1.9.0 | Local literature similarity search |
| `requests` | 2.32.5 | Web, PMC, and Vertex Search requests |
| `beautifulsoup4` | 4.13.5 | HTML and PMC parsing |
| `biopython` | 1.85 | NCBI Entrez access |
| `openpyxl` | 3.1.5 | Excel generation and formatting |
| `XlsxWriter` | 3.2.6 | Excel charts and preferred report export |
| `sentence-transformers` | 5.1.0 | Local text embeddings |
| `torch` | 2.8.0 | SentenceTransformer runtime |
| `transformers` | 4.56.1 | SentenceTransformer runtime |
| `openai` | 1.108.0 | OpenAI API and Azure OpenAI clients |
| `azure-ai-inference` | 1.0.0b9 | Azure inference client used by Skill 1 |
| `azure-core` | 1.35.1 | Azure credentials used by Skill 1 |
| `PyQt5` / Conda `pyqt` | 5.15.11 | Desktop GUI |
| `PyQtWebEngine` / Conda `pyqtwebengine` | 5.15.7 / 5.15.11 | Embedded HTML report viewer |

`sentence-transformers` also installs packages such as SciPy, scikit-learn, Hugging Face Hub, tokenizers, and tqdm. The verified environment used SciPy 1.15.3, scikit-learn 1.7.2, Hugging Face Hub 0.35.0, tokenizers 0.22.1, and tqdm 4.67.1.

`azure-ai-inference` and `azure-core` are currently required even for OpenAI-only use because Skill 1 imports them when the module starts. `PyQt5` and `PyQtWebEngine` are required for the GUI but can be omitted for command-line-only use.

No MySQL, PostgreSQL, SQLite, SQLAlchemy, or other database server/driver is required. AgentLnc uses the files in `database/`, Python's standard library, and FAISS.

### Recommended installation

Conda or Mamba is recommended, particularly on Apple Silicon, because FAISS and Qt WebEngine contain native components.

```bash
conda create -n agentlnc -c conda-forge \
  python=3.10.18 \
  numpy=1.26.4 \
  pandas=2.3.2 \
  faiss-cpu=1.9.0 \
  requests=2.32.5 \
  beautifulsoup4=4.13.5 \
  biopython=1.85 \
  openpyxl=3.1.5 \
  xlsxwriter=3.2.6 \
  pyqt=5.15.11 \
  pyqtwebengine=5.15.11

conda activate agentlnc

python -m pip install \
  "torch==2.8.0" \
  "transformers==4.56.1" \
  "sentence-transformers==5.1.0" \
  "openai==1.108.0" \
  "azure-ai-inference==1.0.0b9" \
  "azure-core==1.35.1"
```

The versions above reproduce the tested environment. Newer compatible versions may work but have not been validated with this release.

Verify the installation:

```bash
python -c "import numpy, pandas, faiss, requests, bs4, Bio, openpyxl, xlsxwriter, sentence_transformers, openai, azure.ai.inference; from PyQt5.QtWebEngineWidgets import QWebEngineView; print('AgentLnc dependencies are available')"
python -m pip check
```

On first use, SentenceTransformers may download `sentence-transformers/all-MiniLM-L6-v2`. Internet access or an existing Hugging Face cache is therefore required. On Linux, the GUI also requires a working desktop session and the Qt/OpenGL/XCB system libraries supplied by the operating system.

## What AgentLnc does

### Skill 1 - lncRNA-RBP inference

Skill 1 combines NPInter, starBase, RNAInter, a local FAISS literature index, PubMed/PMC, and optional user files. It identifies candidate lncRNA-binding RBPs and produces functional reasoning, evidence checks, a knowledge graph, an HTML report, and an Excel supplement.

### Skill 2 - RBP evidence check

Skill 2 searches for literature related to an RBP and a research question, filters candidate articles, reads PMC full text, extracts conclusions and experiments, grades evidence, and generates HTML and Excel reports.

### Skill 3 - phenotype inference

Skill 3 combines an RBP, an optional lncRNA, GTEx tissue expression, RBPbase/GWAS, Gene_Assay, BioGRID/HINT PPI, optional liver DRS data, and PMC evidence. It creates one integrated hypothesis per selected regulation direction. `all` creates one up-regulation and one down-regulation hypothesis.

## Primary project layout

```text
AgentLnc/
|-- README.md
|-- tool_GUI.py
|-- skill1_lncRNA_RBP_inference.py
|-- skill2_RBP_evidence_check.py
|-- skill3_phenotype_inference.py
|-- local_db_handler.py
|-- profile.json
|-- database/
|   `-- DRS_regulation/
|-- icons/
|-- snapshot/     # Shipped demo snapshots and snapshots created by later runs
`-- temp/         # Shipped demo reports and newly generated HTML/XLSX files
```

The main AgentLnc application consists of `tool_GUI.py` and the three Skill scripts. `local_db_handler.py`, `database/`, and `icons/` are support components and must remain next to the scripts.

## API and profile configuration

Each Skill loads a valid JSON profile before initializing its API clients. By default, the file is `profile.json` next to the scripts. Set `PROFILE_JSON` to use a different valid JSON file.

The provider selection rule is:

1. If `OPENAI_API_KEY` is non-empty, AgentLnc uses the OpenAI API.
2. Otherwise, AgentLnc uses the configured Azure endpoints and keys.

A profile template is shown below. Replace placeholders locally and never publish real credentials.

```json
{
  "OPENAI_API_KEY": "",
  "OPENAI_GPT_MODEL": "gpt-4o",
  "OPENAI_REASONING_MODEL": "o4-mini",

  "AZURE_GPT4O_ENDPOINT": "",
  "AZURE_GPT4O_API_KEY": "",
  "AZURE_O4MINI_ENDPOINT": "",
  "AZURE_O4MINI_API_KEY": "",

  "VERTEX_PROJECT": "",
  "VERTEX_ENGINE_ID": "",
  "VERTEX_API_KEY": "",
  "VERTEX_LOCATION": "global",
  "VERTEX_COLLECTION": "default_collection",
  "VERTEX_SERVING_CONFIG": "default_search",

  "ENTREZ_EMAIL": "your.email@example.org"
}
```

OpenAI credentials can also be supplied through environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_GPT_MODEL="gpt-4o"
export OPENAI_REASONING_MODEL="o4-mini"
```

Keep the corresponding keys in `profile.json` when environment variables are used. Same-name environment variables override values stored in the JSON profile.

OpenAI replaces only the LLM provider. It does not replace the Vertex AI Discovery Engine/searchLite configuration used for fresh literature retrieval. `ENTREZ_EMAIL` should contain a valid contact email for NCBI requests. Skill 1 also accepts an optional `NCBI_API_KEY` environment variable.

Even snapshot replay initializes the configured LLM client when the script starts, so either OpenAI or Azure credentials must still be available. An exact matching snapshot avoids new search and LLM calls after startup.

> `profile.json` contains credentials. Do not commit it to a public repository, send it with reports, or share it with untrusted users.

## GUI quick start

Run the GUI from the project directory with the same Python interpreter in which the dependencies were installed:

```bash
cd /path/to/AgentLnc
python tool_GUI.py
```

The GUI provides multiple Agent tabs, the three Skill selectors, structured progress output, raw logs, an embedded HTML report viewer, and system opening for Excel files. Up to four Agent runs can execute concurrently.

Each Skill page has a prominent `DEMO` button. It fills the tested parameters associated with the shipped snapshot and output files. The button only fills the form; it does not start a run. Review the values and click `Run` separately. Demo values use Fresh Mode `N` so an exact matching shipped snapshot can be replayed.

## Tested command-line demos

Run command-line Skills from the AgentLnc root directory. This is required for Skill 1 and Skill 2 because their `temp/` output path is relative to the current working directory.

### Skill 1 demo

```bash
python skill1_lncRNA_RBP_inference.py \
  --gene_of_interest "PRKAG2-AS1" \
  --user_query "I want to know the binding protein of PRKAG2-AS1 to inference its potential function in liver metabolism" \
  --reasoning_function "Liver metabolism" \
  --binding_database "NPInter,starBase,RNAInter" \
  --literature_searching "Similarity,PubMed" \
  --output_file_name "PRKAG2-AS1_R1" \
  --quotauser "quota0001" \
  --entity_type lncrna \
  --fresh N
```

Do not add `--external_information` to this demo command, because the shipped Skill 1 snapshot was created without external files.

Skill 1 required arguments are `--gene_of_interest`, `--user_query`, `--reasoning_function`, `--binding_database`, and `--literature_searching`.

Optional arguments:

- `--external_information`: one or more file specifications separated by `|`; each item may include `(description: ...)`.
- `--output_file_name`: report name without an extension; a name is generated automatically when omitted.
- `--quotauser`: search quota identifier.
- `--entity_type`: `lncrna`, `rbp`, or `mrna`; default `lncrna`.
- `--fresh`: `Y` or `N`; default `N`.

Skill 1 writes `./temp/<name>.html` and `./temp/<name>.xlsx` relative to the current working directory.

### Skill 2 demo

```bash
python skill2_RBP_evidence_check.py \
  --query "liver metabolism" \
  --gene "RBFOX2" \
  --output "RBFOX2_liver_metabolism" \
  --fresh N
```

`--query`, `--gene`, and `--output` are required. `--fresh` accepts `Y` or `N` and defaults to `N`. Skill 2 writes `./temp/<name>.html` and `./temp/<name>.xlsx` relative to the current working directory.

### Skill 3 demo

```bash
python skill3_phenotype_inference.py \
  --rbp "RBFOX2" \
  --lncrna "PRKAG2-AS1" \
  --tissue "Liver" \
  --regulation_type all \
  --append_info "We confirmed PRKAG2-AS1 could bind to RBFOX in human and mouse liver tissue." \
  --output "PRKAG2-AS1_RBFOX2_liver_metabolism" \
  --fresh N
```

`--rbp`, `--tissue`, and `--output` are required. `--lncrna` and `--append_info` are optional. `--regulation_type` accepts `all`, `up_regulation`, or `down_regulation` and defaults to `all`. Skill 3 writes `temp/<name>.html` and `temp/<name>.xlsx` next to the script.

## Fresh runs, snapshots, and outputs

- `--fresh N`: use an exact matching snapshot when one exists; otherwise perform a fresh analysis.
- `--fresh Y`: ignore an existing snapshot, perform the full analysis, and save a new snapshot.
- Skill 1 snapshot matching includes hashes of external-file contents.
- `snapshot/` contains gzip-compressed JSON snapshots keyed by normalized inputs.
- `temp/` contains HTML reports and Excel supplements.
- Fresh runs require network access to the configured APIs and external literature sources.

## Data and security

- Use the bundled, trusted files in `database/`. `meta.pkl` is a Python pickle file and must not be replaced with a file from an untrusted source.
- Do not publish `profile.json`, private research inputs, generated reports, or snapshots that contain sensitive information.
- The local database handler does not connect to an external database server.
- API availability, rate limits, and third-party service permissions remain the user's responsibility.


