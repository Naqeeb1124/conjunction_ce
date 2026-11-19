# NOVELTY.md: prior-art checks

Two prior-art checks were run in November 2026. The first was on the original plan and changed the project. The second was on the question the project now asks.

## 1. Summary

**Original plan:** chance-constrained correlated-equilibrium (CC-CE) recommendations for satellite conjunctions, presented as a new application.

**What the first check found:** no published paper does it, but Jaehan Im, first author of all four CC-CE papers, writes on his research page (updated Sep 2026) that he is studying "chance-constrained correlated recommendations for noncooperative spacecraft collision avoidance". One sentence, no model, no manuscript found. Source: https://sites.google.com/view/jaehanim/research. Building the same thing would have duplicated work in progress in the group the method comes from, so the plan was dropped.

**Current question:** a coordinator's recommendation works only if operators know the rule it is drawn from. If they have to learn the rule from the events they have seen, how do a fixed rule, public recommendations and private (correlated-equilibrium) recommendations compare as the number of past events grows, and what margin on the incentive constraints is best?

**What the second check found:**

- The learning question has been studied for **one** follower or **one** receiver, including by the groups whose papers are listed in section 2.1. Nothing was found for a committed correlated equilibrium, or for several receivers with private signals.
- Two effects this project shows for two receivers have published single-receiver counterparts, and the sample bound it uses is a standard argument. They must be presented as counterparts, not as new techniques.
- No work was found on how many interactions satellite operators need before a coordination service's recommendations can be learned.

**What can be said:** "as far as a search on 18 Nov 2026 could find, the inferability question has not been put to a correlated-equilibrium coordinator with two receivers, and not in space traffic coordination." **What cannot be said:** that the binding-constraint effect or the margin-squared sample bound are new, that correlated equilibria for conjunctions are a new idea, or that nobody is working on them.

## 2. Prior art for the current question

### 2.1 Learning a commitment from samples (one follower or receiver)

| Work | What it does | How this project differs |
|---|---|---|
| Karabag, Smith, Mehr, Fridovich-Keil, Topcu, "When Should a Leader Act Suboptimally? The Role of Inferability in Repeated Stackelberg Games", arXiv:2310.00468 (abstract read; IEEE TAC 2026 per the search notes) | A leader plays a fixed mixed strategy; one follower infers it from past actions. Defines the inferability gap and bounds it by the number of interactions and the stochasticity of the strategy. | Two receivers, and what must be inferred is the correlation between their private recommendations, not one player's mixed strategy. |
| Probine, Karabag, Topcu, "Designing Inferable Signaling Schemes for Bayesian Persuasion", arXiv:2510.01434 (abstract read; body not read in full) | One sender, one receiver who infers the signaling scheme. Loss grows with the number of signals and shrinks as the receiver's best actions become more distinct; schemes that are optimal under known commitment do poorly for every k. | No payoff state here and two receivers playing a game with each other. The same qualitative effects appear (sections 4.2 and 5 of RESULTS.md). |
| Sood and Best, "Frequentist Persuasion", arXiv:2510.25066 | One receiver forms beliefs from K samples. At the classical optimum the probability that a recommendation is credible tends to one half. | The one-receiver counterpart of DERIVATIONS.md 5.2. |
| Zhang and Li, arXiv:2608.20766 | One receiver; with an obedience margin, a sample bound of order 1 / (smallest signal probability x margin squared). | The one-receiver counterpart of the bound in DERIVATIONS.md 5.3. |
| Lin and Chen, arXiv:2402.09721; "Calibrated Stackelberg Games", arXiv:2306.02704 | One agent who learns the principal's strategy by no-regret or calibrated forecasting. | One agent; different learning model. |

### 2.2 Samples from a correlated equilibrium; mediators with partly known devices

| Work | What it does | How this project differs |
|---|---|---|
| Babichenko, Barman, Peretz, "Empirical Distribution of Equilibrium Play and Its Testing Application", arXiv:1310.7654 | If players follow a correlated equilibrium, the empirical distribution of k samples is an approximate correlated equilibrium once k is of order (m ln m + ln n) / eps^2; Hoeffding plus a union bound. | Same argument as DERIVATIONS.md 5.3. Theirs is from an observer's side with an ex-ante tolerance; here it is per recommendation, with a margin the coordinator chooses, and the object of interest is the coordinator's realised cost against k. |
| Giordano, "Coordination Mechanisms with Partially Specified Probabilities", arXiv:2605.07469 | Several players and a mediator whose device is only partly disclosed; players complete their beliefs by a fixed rule. Lists learning from past play as open. | No samples there; here beliefs are noisy frequencies from k events. |
| Hines and Larson, arXiv:1206.3261 | Players run a statistical test on whether others follow an announced mediator. | The device is known there. |
| McMahan et al., "Optimally Installing Strict Equilibria", arXiv:2503.03676 | Designs rewards so that a target is a strict correlated equilibrium. | Strictness is for bounded rationality, with the rule known. |
| Bone, Drouvelis, Gurguc, Ray, Cardiff Economics Working Paper E2024/27 | Laboratory experiment in a game of chicken: public against private correlation devices, announced to the subjects. | The only public-versus-private comparison found; behavioural, and the devices are announced, not learned. |

The items in 2.1 and 2.2 other than the first two were found in the prior-art search and are described from search notes; they were not read in full for this write-up.

### 2.3 The domain

Nothing was found that counts the interactions operators need to learn or trust a coordination service, or that discusses the learnability of right-of-way rules. Related material: simulations of candidate right-of-way rules across the catalogue (Borowitz et al., AMOS 2021 and IAC 2024); rule engines and negotiation platforms (section 3.3); studies of alarm fatigue from conjunction warnings (Wiley, Roy, Anderson, IAC 2020).

## 3. Prior art for the original question

### 3.1 The CC-CE line (same method, other domains)

| Work | What it does |
|---|---|
| Im, Topcu, Fridovich-Keil, "Chance-Constrained Correlated Equilibria for Robust Noncooperative Coordination", arXiv:2603.14141, IEEE L-CSS 10 (2026) 1273-1278 | Defines CC-CE. One chance constraint per incentive constraint at one confidence level; one additive Gaussian term per agent on its deviation cost, which keeps the program linear; no sampling in the optimisation; agents are assumed to know the recommendation distribution. Vertiport occupancy game, compared with the nominal CE only. Names non-Gaussian and deviation-specific uncertainty, and feasibility under large uncertainty, as future work. |
| Im, Topcu, Fridovich-Keil, arXiv:2609.13565 | Exact-one resource-assignment restriction; the restricted feasible set is the convex hull of chance-constrained pure Nash equilibria. Vertiport departure corridors. (Not read in full here; described from search notes.) The scale-up of RESULTS.md 4.10 borrows only the restriction: exactly one satellite burns in each conjunction. In that paper's setting agents choose among shared resources, while here every conjunction is its own two-sided game with three burn levels and no margin, so the counterpart of the hull result is stated and proved again for this model (DERIVATIONS.md 7.4) rather than taken over. The scale-up adds no method of its own: it uses that restriction, standard linear programs and the fact that costs here add up over conjunctions. |
| Im, Fridovich-Keil, Topcu, arXiv:2604.00456 and arXiv:2602.14436 | Scalable CC-CE as convex combinations of chance-constrained pure Nash equilibria; multi-airline and airport-surface experiments. |
| Im, "Noncooperative Coordination for Decentralized Air Traffic Management", arXiv:2602.15333 | A short single-author overview of the same research programme for air traffic management. Its abstract does not mention space. |
| Im's research page | One sentence stating ongoing work on the spacecraft application (section 1). |

In this project the CC-CE deterministic equivalent appears as the same margin on every constraint (DERIVATIONS.md 4.6). The convex-hull result has a reading here too: a mixture of pure Nash equilibria is what a *public* recommendation can achieve (DERIVATIONS.md 4.5).

### 3.2 Game-theoretic models of the two-operator conjunction decision

| Work | What it does | Relation |
|---|---|---|
| Dolan, Qin, Ding, Balakrishnan, "Satellite Collision Avoidance Using Repeated Games", AAS 23-302 (2023) | Two operators, move or wait at hourly steps, framed as a multi-step game of chicken. Pure and mixed Nash equilibria; a decision tree over several steps; a second model with private risk types and thresholds. Foster-Estes collision probability. | No third party, no correlation device, no incentive constraints from a coordinator's side. |
| Dolan, PhD thesis, MIT, Feb 2025, Chapter 5, pp. 61-76 (read in full from the extracted text) | Thesis version. Types are known to the other satellite. Mixed-equilibrium probability of moving rises with the ratio of collision cost to maneuver cost; with more decision steps satellites move earlier. | The source for the "no coordinator" baseline here: a mixed equilibrium with a positive chance that neither moves. |
| Venigalla and Scheeres, AIAA SciTech 2022, doi:10.2514/6.2022-1274 | Pareto front for cooperative avoidance; local Nash equilibria of a deterministic differential game, computed with the iterative linear-quadratic solver of Fridovich-Keil et al. Finds several equilibria and leaves the choice open. | Deterministic, no collision probability, no coordinator. The selection problem it leaves open is the one a coordinator addresses. |
| Hayhurst and Colombi, "Game-Theoretic System Design in the Development of Space Power", Air and Space Power Journal 35(2), 2021 | A qualitative list of 2x2 games with space examples. Names a correlated equilibrium as a natural way to resolve collision-avoidance maneuvering, through alternation or a third party. | Words only, nothing computed, but it means the idea of a correlated equilibrium for this problem is not new. A strategy journal outside the civil-coordination literature. |

### 3.3 Third parties that decide or negotiate who moves

| Work | What it does |
|---|---|
| Kayhan Space, Satcat "Rules of the Road" (documentation) | Assigns responsibility from maneuverability categories; advisory; equal categories fall back to bilateral coordination. A separate engine optimises the burn. This is the kind of fixed rule this project uses as its baseline. |
| ESA CREAM, GMV "AutoSTM" platform (ESA Clean Space Industry Days 2024; IAC-24,A6,IPB,32) | Rule filter, then automated negotiation between software agents, then mediation by an authority on request. |
| OKAPI:Astrolabe (SpaceOps 2025); ISO/TC 20/SC 14 committee draft 9490 | A rule engine with baseline, global and bilateral protocols; a draft default table of who gives way. |
| Hendriks et al., "Starling 1.5", SmallSat 2025 | On-orbit demonstration in which operators claim or refuse responsibility through a screening hub. |
| Space Guardian; Satellite Negotiator (company pages) | An auction with a payment to the operator that moves; a planned product with bargaining software agents. No published mechanism. |
| Nag, Murakami, Lifson, Kopardekar, DASC 2018; Nag et al., Acta Astronautica 2021 | Architecture papers with a prose list of ways to decide who moves: right-of-way rules, cost split, auction, lower-cost operator moves, last-minute. None is modelled. |

### 3.4 Cooperative planning and optimisation

| Work | What it does |
|---|---|
| Kim, Al-Husseini, Eddy, Kochenderfer, arXiv:2607.26570 | Two satellites share one team reward; a semi-decentralized POMDP handles intermittent ground contact. No private costs, no equilibrium, no collision probability. Complementary: it asks how much communication a cooperative team needs. |
| Rubio, Biondi, Escobar, Di Lizia, IAC 2024, doi:10.52202/078360-0163 | One optimisation over both satellites' burns with a fairness split fixed in advance; agreement is assumed. |

### 3.5 Correlated-equilibrium coordinators in neighbouring traffic domains

Tony, Ghose, Chakravarthy (J. Aerospace Information Systems 2022, doi:10.2514/1.I011001) for small-aircraft conflict resolution; Wang, Yi, Hong (arXiv:2406.05336) for road intersections; Im et al. for air traffic management. None concerns satellites.

## 4. What this project adds

Shown by the runs and tests in this repository (RESULTS.md). All of it is for one baseline conjunction and its sampled variations; the numbers are model results.

1. A worked, reproducible two-operator conjunction game with tested physics: 2D collision probability, a burn-to-miss mapping, three burn levels.
2. The inferability question for a correlated-equilibrium coordinator with two receivers: realised coordinator cost and obedience as functions of the number of past events, for public rules, the textbook private rule and private rules with a margin. The simulation is checked against an exact enumeration of the learning histories.
3. A crossover between public and private recommendations in the number of past events. With costs known and operators who start from one observation of each recommendation, the private rule with a targeted margin beats the public rule from 13 past events in the baseline, and the rule with the same margin on every constraint from 96. The 13 depends on how operators treat a recommendation they have barely seen: if an unseen recommendation leaves a uniform belief, the targeted rule is ahead from the first event. When the coordinator knows the costs to about 10%, the gain is about 20% after a few hundred events; to about 30%, the best private rule is only level with the public rule.
4. Where the margin is needed. For learnability it is needed only on the constraints on which an operator is unsure what the other was told. The same margin on every constraint, which is the CC-CE form, is capped in the baseline at 1.886e-4 (1.8e-4 on the grid used) by a constraint whose sign does not depend on what the operator believes; the targeted margin reaches 1.04e-3 on its grid.
5. Two negative results: the plain scenario program is infeasible at the baseline uncertainty, with the cause exhibited (six obedience constraints from three sampled games that no recommendation table satisfies together); and a violation budget chosen for low nominal cost gives a high realised cost.

Items 2 to 4 are the part not found elsewhere. That is a statement about a limited search, with the limits below. No new technique is claimed: the failure of the textbook rule is the two-receiver counterpart of the binding-constraint effect known for one receiver (Sood and Best; Probine, Karabag and Topcu), and the sample bound with the margin squared is the standard Hoeffding-plus-union-bound argument (Babichenko, Barman and Peretz; Zhang and Li for one receiver).

## 5. How the checks were done, and their limits

- **First check.** The three queries set at the start of the project, each with ten or more variants, on arXiv and on journal indexes; six search angles (arXiv, journals, conference proceedings and agency reports, mechanism design and economics, citation chasing, other solution concepts and neighbouring domains); three further searches that assumed the work exists. About 70 candidate entries, several of them the same work under different titles; sixteen follow-up reads covering nine distinct items. Chapter 5 of Dolan's thesis was read in full from MIT's extracted-text file.
- **Second check.** 33 queries across three search tools on learning a mediator's or sender's commitment from samples, margin and strict correlated equilibria, multi-receiver persuasion with learning receivers, and learnability of coordination rules in space and aviation.
- **Limits.** Search indexes cover late-2026 conference papers poorly; a paper under review would not be found. Citation graphs were not available, so citing papers were searched by phrase. Google Scholar, SSRN and 2026 conference programmes were not searched systematically. Many items were read as abstracts only. Two search services ran out of quota part-way through the first check.

## 6. Sources

- Im, Topcu, Fridovich-Keil: arXiv:2603.14141, 2609.13565, 2604.00456, 2602.14436. J. Im, research page: https://sites.google.com/view/jaehanim/research
- Im, J., "Noncooperative Coordination for Decentralized Air Traffic Management", arXiv:2602.15333 (abstract read)
- Karabag et al.: https://arxiv.org/abs/2310.00468. Probine, Karabag, Topcu: https://arxiv.org/abs/2510.01434
- Sood, Best: https://arxiv.org/abs/2510.25066. Zhang, Li: https://arxiv.org/abs/2608.20766. Lin, Chen: https://arxiv.org/abs/2402.09721. Calibrated Stackelberg Games: https://arxiv.org/abs/2306.02704
- Babichenko, Barman, Peretz: https://arxiv.org/abs/1310.7654. Giordano: https://arxiv.org/abs/2605.07469. Hines, Larson: https://arxiv.org/abs/1206.3261. McMahan et al.: https://arxiv.org/abs/2503.03676. Bone et al.: http://carbsecon.com/wp/E2024_27.pdf
- Dolan, Qin, Ding, Balakrishnan, AAS 23-302: https://www.mit.edu/~hamsa/pubs/Dolan-etal-AAS2023.pdf. Dolan, PhD thesis: https://hdl.handle.net/1721.1/158894
- Venigalla, Scheeres: https://doi.org/10.2514/6.2022-1274. Kim et al.: https://arxiv.org/abs/2607.26570
- Hayhurst, Colombi: https://www.airuniversity.af.edu/Portals/10/ASPJ/journals/Volume-35_Issue-2/F_Hayhurst_Colombi.pdf
- Nag et al.: https://ntrs.nasa.gov/citations/20180007349 and https://doi.org/10.1016/j.actaastro.2020.11.056. Rubio et al.: https://doi.org/10.52202/078360-0163
- Kayhan Space, Satcat documentation: https://docs.satcat.com/key-concepts/rules-of-the-road/. GMV / ESA, AutoSTM: https://indico.esa.int/event/516/contributions/10024/. OKAPI:Astrolabe: https://publications.spaceops.org/2025/download_by_id.php?id=0059. Starling 1.5: https://doi.org/10.26077/40d0-4262
- EUSPA on Space Guardian: https://www.euspa.europa.eu/newsroom-events/success-stories/space-guardian-automating-collision-avoidance-safer-orbit. Satellite Negotiator: https://satellitenegotiator.com/
- ISO/TC 20/SC 14 committee draft 9490 (read in part): https://www.space.commerce.gov/wp-content/uploads/2025/04/ISO-TC-20-SC-14_N2559_CD9490_USTAG.pdf
- Borowitz et al.: https://amostech.com/TechnicalPapers/2021/Conjunction-RPO/Borowitz.pdf and https://ntrs.nasa.gov/citations/20240011833
- Wiley, Roy, Anderson, paper IAC-20,A6,7,6,x59534, International Astronautical Congress 2020, on alarm fatigue for operators receiving conjunction warnings (abstract only read): https://p.iafastro.directory/iac/paper/id/59534/abstract-pdf/IAC-20,A6,7,6,x59534.brief.pdf
- Tony, Ghose, Chakravarthy: https://doi.org/10.2514/1.I011001. Wang, Yi, Hong: https://arxiv.org/abs/2406.05336
