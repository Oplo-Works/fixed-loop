# Small Model, Fixed Loop, Measured Iterations: Post-Training Gemma 4 E4B into an Offline Coding Agent

## Abstract

This paper designs an offline coding agent by post-training Gemma 4 E4B for the Paper Track of the Gemma 4 Developer Agent Competition on Kaggle. The agent resolves hidden Python bug fixes and feature requests offline: given a repository and an issue description, it produces a patch scored by hidden tests. The metric, here called Resolution Rate, is the fraction of approximately 120 hidden tasks resolved within 12 hours. Three claims are central. First, the harness rewards a narrow behavior — patches that pass hidden tests under a hard time budget — and the design is derived from that; minimality is the author's heuristic, not the reward itself. Second, under the author's constraints (solo, first-time ML practitioner, Kaggle free GPU only, 21 days), E4B is the author's best judgment of the largest model trainable and resubmittable in time — a hypothesis. Third, the process matters as much as the model: pilot-first training, a failure taxonomy for local evaluation, and one stated change per daily submission. The build has not run yet; quantities to be measured are marked [TO BE MEASURED].

## 1. Introduction

The strongest coding agents today depend on three things this competition removes: a large proprietary model, persistent internet access, and effectively unlimited compute. Without them, the question is concrete: what can a small open model, post-trained by one person on free hardware, actually do? The Gemma 4 Developer Agent Competition asks exactly that: turn an open Gemma 4 model into an autonomous offline coding agent whose patches pass hidden tests.

This paper answers with a design, not a result. The author is a solo participant — an application developer with no ML training experience, no local GPU, and only Kaggle's free GPU tier — aiming to learn and to field a working leaderboard entry. Those facts are the design constraints: every choice is justified against them, and compromises are marked.

## 2. Related Work

The evaluation protocol this competition adopts comes from SWE-bench (Jimenez et al., 2024), which established test-based resolution of real GitHub issues as the standard measure: a patch counts only if failing tests pass and passing tests do not regress. SWE-agent (Yang et al., 2024) showed an agent-computer interface can resolve such issues autonomously, using large proprietary models under a per-instance API budget. Agentless (Xia et al., 2024) showed that a fixed three-phase pipeline — localization, repair, validation — can rival autonomous agent loops on SWE-bench, which supports this paper's fixed-loop design. LoRA (Hu et al., 2022) trains small adapter weights on a frozen base model, and QLoRA (Dettmers et al., 2023) extended that to quantized LLMs on a single 48 GB GPU. SWE-Gym (Pan et al., 2025) provides 2,438 executable Python tasks and released agent trajectories, the closest template for the trajectory construction in Section 5. Gemma 4 (Google DeepMind, 2026) provides the open weights this competition builds on. What prior work does not address is this combination: a small open model, an offline sandbox, a declarative-only agent, and a single first-time practitioner on free hardware.

## 3. Harness and Constraints Analysis

The evaluation is SWE-bench style: the agent's patch is applied at the task's base commit, the official test patch is applied, and hermetic pytest decides pass or fail. A task counts as resolved only if failing tests now pass and passing tests still pass. Resolution Rate is the fraction of approximately 120 hidden tasks resolved, evenly divided between public and private splits and curated from private repositories. The agent has 12 hours for all tasks, inclusive of sandbox setup and excluding patch validation.

Two numbers derived from the harness drive the whole design. First, 12 hours for ~120 tasks is an average of about 6 minutes per task, including inference, tool use, and test execution. Time discipline is therefore not an optimization; it is a precondition for attempting the full set. Second, the outcome per task is binary: an almost-correct patch earns nothing, so the agent must be built around verification — but verification ranks patches, it does not gate them.

The submission is a submission.zip with a declarative YAML configuration (agent.yaml at the root, no agent.py); this design submits PEFT LoRA adapters in .safetensors format under adapters/<name>/. Only one submission per day is allowed, with up to two final submissions. The runtime sandbox is air-gapped: no internet, and setup installs offline from 124 bundled wheels. HARNESS_README.md further specifies a maximum LoRA rank of 128, a 3 GiB submission cap with no pickle files, and a 4 GiB RAM / 2 vCPU sandbox.

Three requirements follow. (1) The agent must be small and self-contained: a fixed loop, bounded retries, local verification, nothing outside the submission. (2) Development must be measurable locally, because each leaderboard submission is a scarce experiment. (3) The process must be recoverable: a failed run or bad submission must cost hours, not days.

## 4. Model Selection: Why Gemma 4 E4B

The base model was chosen by elimination under the author's constraints, not by benchmark comparison. The competition permits four quantized Gemma 4 variants — 31B, 12B, E4B, E2B, all QAT w4a16 — and the $10,000 small-model bonus goes to the highest-ranking E2B or E4B submission, which must be one of the team's two final selections.

The 31B default is eliminated on iteration cost, not memory: it is the largest variant, and competition notebooks offer L4x4 machines (4 × 24 GB) at twice the quota rate, so 31B training is likely feasible. But at a ~30-hour weekly quota, L4x4 use leaves about 15 wall-clock hours a week, which a first-time practitioner debugging 31B training would spend on infrastructure, not iteration. The risk is an unsubmitted entry, not an untrainable model.

The 12B model is eliminated on iteration speed: plausibly trainable, but not reliably within session limits, a ~30-hour weekly quota, and a first-time practitioner's debugging speed. One failed multi-hour run would consume a large share of the week's budget with nothing to show for it.

E2B is set aside on a capability hypothesis: fixing hidden bugs needs sustained reasoning — holding the issue, the relevant code, and test output in context, forming a hypothesis, editing precisely, verifying. A model too small to sustain that loop may not be rescuable by prompting or scaffolding. E2B remains the fallback.

E4B remains the author's best judgment of the largest model iterable within the quota: small enough to train and evaluate within session limits (4.5B effective parameters, 128K context, per the Gemma 4 model card), yet large enough for a plausible chance on bug-fixing tasks, and bonus-eligible.

This is a deliberate compromise: E4B's ceiling is below the 12B and 31B models, and some hidden tasks will be beyond its reach regardless of training. The final Resolution Rate is [TO BE MEASURED].

## 5. Post-Training Design

Training uses LoRA with rank at most 128, the harness limit; the planned starting point is rank 32 on attention and MLP modules. LoRA keeps memory and time within free-GPU limits and produces a small adapter (adapter_config.json plus adapter_model.safetensors) within the submission cap.

Training data will be public SWE-bench-style data (source not yet selected): issue descriptions with repository context, gold patches, and test outcomes. Gold patches contain no trajectories, so trajectories must be constructed. The likely primary source is scripted reproduce–localize–patch–verify traces built around gold patches, replayed in task environments to record real tool outputs; gold patches supervise edits, not agent-visible context. Rejection sampling — running the untrained base E4B agent on training tasks and keeping only runs whose patch passes the tests — is a supplement, and an expensive one: 300 attempts at about 6 minutes each cost about 30 wall-clock hours (60 quota-hours; roughly two weeks). The pilot must validate training support and adapter compatibility with the served QAT w4a16 checkpoint before longer training. Training tasks will be checked against the four public development repositories (FastAPI, Requests, Rich, HTTPX) and any overlap removed, so local scores are not inflated.

Filtering matters more than volume under a ~30-hour weekly budget: the training set will exclude unclear issues, patches touching many unrelated files, missing or non-deterministic tests, and anything needing network access at run time. The number of training examples, steps, learning rate, and final loss will be recorded after training as [TO BE MEASURED].

The objective is behavioral, not merely textual: gather evidence before editing, stop exploring when there is enough, produce patches that apply cleanly, run tests before declaring success. Examples showing only a final patch, without the preceding reasoning and tool use, would teach guessing. Training loss is recorded for diagnosis only: loss can fall while behavior stays poor, if the model learns plausible-looking patches that fail tests. The relevant checks are behavioral, as described in Section 7.

The sequence is pilot-first: a first run on a small subset for about 30 minutes verifies the full pipeline end to end — data loading, training, adapter saving and loading, and a local agent run. Only then will a longer run be attempted; a long first run would cost a day if a formatting error surfaced late, the pilot costs 30 minutes. Checkpoints will be saved frequently so a session timeout loses minutes, not a run. Data, configuration, and adapter will be versioned together, so any submitted adapter is traceable to what produced it.

## 6. Agent Design

The agent is specified declaratively in YAML, with no agent.py. The YAML defines the model's role, the steps, and the conditions for stopping, retrying, or abandoning.

The core loop has four stages: reproduce, localize, patch, verify. In reproduce, the agent reproduces the fault with the smallest relevant test or a minimal script in /tmp via run_command, recording the failure message. If it cannot reproduce the failure, it says so in its trace, gathers more context within a bounded search, and proceeds on the issue text alone, marking any resulting patch as unverified. For feature requests, with no failure to reproduce, this stage becomes a minimal demonstration of the missing behavior. In localize, the agent narrows the fault to specific files and functions, using failing-test symbol names with search_similar_code and get_code_neighbors before read_file, and forms a single hypothesis, preferring the simplest explanation. In patch, it makes a minimal edit via edit_file: only the code necessary to fix the reported issue, preserving signatures and public behavior. In verify, it re-runs the reproduction plus the relevant existing tests via run_command. "Verification" means the agent's own reproduction and existing tests; it cannot run the hidden validation tests.

The stop policy is single: submit the best patch that applies cleanly, even if unverified; leave the patch empty only if nothing applies. Candidate diffs are saved in /tmp and ranked by reproduction, then relevant tests, then size; an applicable unverified patch may succeed, an empty patch offers no attempted fix. Retries are capped at k = 2 revised patches; the planned per-task caps are soft 5 minutes (pace checks) and hard 8 minutes (eval_config.yaml). The binding constraint is pace — 120 tasks must average 6 minutes — so the agent reads get_status and abandons long tasks early when behind pace. The agent finishes with submit_patch.

The actions are predefined harness tools, not skills: read_file, edit_file, write_file, run_command, submit_patch, get_status, and the graph tools get_code_neighbors, search_similar_code, get_code_subgraph. Skills are directories with a SKILL.md manifest; the first version uses them for guidance only, with all actions through the harness tools. Context is managed explicitly so long logs do not crowd out the issue and the current hypothesis. No step calls the internet or assumes a tool absent from the sandbox; if a tool is unavailable, the agent falls back to a simpler alternative.

## 7. Evaluation Plan

Evaluation has two parts — local evaluation before submission, and leaderboard iteration after — with one definition of success: a patch passing hermetic pytest after the test patch is applied. Three data splits are kept disjoint: (i) training data; (ii) a local sample of n = 30 fixed public task IDs, for day-to-day measurement; (iii) a selection sample of n = 30 fixed task IDs, used only to accept or reject a change.

Local evaluation uses the local n = 30 sample from the 129 public development tasks. The first local run uses the untrained base model, establishing a baseline and verifying the measurement method before any training time is spent; the headline planned comparison is adapter-off versus adapter-on. The agent then runs end to end under scoring-like constraints: the same YAML, the same adapter, no internet, a per-task time limit. The primary metric is local Resolution Rate. Secondary measures — patch produced at all, mean time per task, tasks abandoned at the retry limit, patches failing to apply — will be recorded as [TO BE MEASURED]. On n = 30, one task is 3.3 percentage points, so no local score is the hidden score.

The diagnostic core is a failure taxonomy: each failed task receives one label for its earliest blocking cause, with setup and issue defects taking precedence — (a) fault not reproduced; (b) wrong file or function localized; (c) correct hypothesis, incorrect edit; (d) correct edit broken by an unrelated change; (e) verification misleading or skipped; (f) time or retry budget exhausted; (g) environment or setup failure; (h) unclear acceptance criteria. This classification, not the headline rate, decides the next change: localization failure points to search or prompting, verification failure to test selection, budget failure to time limits. A change is accepted only if it gains at least +2 net resolved tasks on the selection sample without increasing mean time per task; repeated reuse makes this validation, not an independent test.

The daily-submission loop is conservative. Before any submission, the package is checked: YAML valid, submission under 3 GiB, each adapter containing adapter_config.json and adapter_model.safetensors with no pickle, no reference to agent.py or missing files. If the check fails, the submission is held and the previous working version stays submitted — a failed package costs a day, so a working submission is always kept ready. Each submission tests one stated change, subject to evaluation budget: at six minutes per task, a 60-task candidate evaluation costs six wall-clock hours (12 quota-hours), excluding startup. Leaderboard feedback covers roughly 60 public-split tasks, so a single day's reading is noisy; a change is judged on the local selection sample, not on one leaderboard move, and a change that does not improve Resolution Rate is not kept. The leaderboard Resolution Rate is [TO BE MEASURED].

## 8. Limitations and Risks

The principal limitation is that this paper reports no results: the design is reasoned from constraints, not tested, and the hidden Resolution Rate is unknown, recorded as [TO BE MEASURED].

The E4B capability ceiling is the main substantive risk: a 4B-class model may struggle with long repository contexts, faults spread across files, and sustained multi-step reasoning. The bounded loop, minimal patches, and verification mitigate this but cannot create capability the base model lacks. Persistent failures may reflect model capacity, training, or scaffolding; this evaluation cannot isolate a model-size ceiling.

Other risks are practical: interrupted sessions, an adapter over the size limit, a YAML referencing an unavailable tool, a missing wheel, a few long tasks consuming the 12-hour budget if per-task limits are not enforced, and a declarative format too restrictive for conditional logic. Finally, overfitting to the local sample: a design that improves locally may not improve on the hidden set. The mitigation is to report the sample's size and procedure and treat the hidden Resolution Rate as the final test.

## 9. Timeline to December 2, 2026

The build runs to December 2 in four gated phases: setup, data, and the untrained baseline (Nov 12–14); pilot and first LoRA run (Nov 15–21); diagnosis and iteration against the Section 7 taxonomy (Nov 22–28); final evaluation and submission of the best measured version (Nov 29–Dec 2). The November 25 entry and team-merger deadline is a hard gate. If a phase slips, the next phase's scope is cut before the submission date moves.

## Conclusion

This paper has presented a design for an offline coding agent built from Gemma 4 E4B: a model chosen by elimination under real constraints, a pilot-first LoRA plan, a declarative YAML agent around a reproduce–localize–patch–verify loop, and a failure-taxonomy evaluation plan. No results are claimed. The next step is to build, measure Resolution Rate [TO BE MEASURED], and let that measurement decide what works.

## References

1. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR. arXiv:2310.06770.
2. Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS. arXiv:2405.15793.
3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
4. Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
5. Google DeepMind (2026). Gemma 4 Model Card. https://gemma.ai/models/gemma-4/model-card/
6. Google DeepMind (2026). The Gemma 4 Developer Agent Competition. Kaggle. https://www.kaggle.com/competitions/gemma-4-developer-agent
7. Google DeepMind (2026). The Gemma 4 Developer Agent Paper Track. Kaggle. https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
8. Xia, C. S., Deng, Y., Dunn, S., and Zhang, L. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
9. Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., and Zhang, Y. (2025). Training Software Engineering Agents and Verifiers with SWE-Gym. ICML. arXiv:2412.21139.
