# How the project works, in plain words

The example numbers on this page are machine-computed by `scripts/worked_example.py`. The
result numbers come from `results/tables/` and `results/runs_*.csv`.

## The problem

A warehouse sends goods to hubs by truck. Each hub sends goods to its own pickup points by van.
Each day we decide which hubs get a truck trip, which points each van visits and in which order,
and how many kg each stop gets. No point may run out, and no stock may pass its storage limit.
The cost is truck trips plus van km plus holding cost (the cost of money tied up in stock).

## The exact model (a mixed-integer program, MIP)

Gurobi solves one model with all decisions. Its rules, numbered as in `src/irp2e/model.py`: a hub
gets goods only on a truck day (1). Stock plus delivery fits the storage limit (2, 4). Stock today
is stock yesterday plus goods in minus goods out, never below zero (3, 5). A point gets at most one
van per day (6). A van enters and leaves each stop once and leaves its hub once (7, 8). A van
delivers only where it stops and carries at most its capacity (9, 10). Each stop gets a position
number that grows along the route, so every route passes the hub (11, the MTZ rule of Miller,
Tucker and Zemlin 1960). Van 2 runs only if van 1 runs (12).

## ALNS (adaptive large neighborhood search)

The search works on a plan: van routes and truck trips, without kg. Each step copies the current
plan, breaks part of it with one of six destroy rules, and fixes it with one of four repair rules.
Every repair adds the visits and trips needed so nothing runs out. What happens to each part:

- Visit days: destroy removes visits. Repair adds a visit between the last visit and the day the
  point would run out.
- Quantities: never searched. Rule J1 (just in time) sets them. Each visit brings what the point
  needs until its next visit, within its storage limit.
- Routes: a new visit goes to the cheapest place in a route with room, or opens a new route if a
  van is free. Then 2-opt reverses a part of the route while that makes it shorter.
- Hub shipments: rule J1 again, on what the hub sends out.
- Truck trips: destroy may remove one. Repair adds one on the first day a hub would run dry.

The search starts from an empty plan, which the greedy repair fills. A cheaper plan is always
kept. A worse plan is kept with a chance that falls over the run (simulated annealing). Rules that
found good plans get a higher weight and are picked more often (Ropke and Pisinger 2006).

## The matheuristic (a heuristic that calls a solver)

Everything is the same as in the ALNS, except the kg. J1 runs first. Only when J1 fails, Gurobi
solves a linear program (LP) for all kg of the fixed plan, at the least holding cost. If the LP
fails too, the plan is rejected. J1 fails when a removal makes an earlier visit carry more and
its van is overloaded. Once the matheuristic keeps a plan that the LP rescued, the plans made from
it fail J1 again until that route changes. When J1 works, the LP cannot do better, because hub
and point holding costs are equal.

## The greedy

Day by day, a point gets goods only when its stock does not cover today. Then it is filled up,
but not above its need until the last day. Vans go to the nearest point that still fits. Hubs
get trucks by the same fill-up rule.

## Worked example: 1 hub, 3 points, 2 days

Hand-made data (`tests/data/tiny.json`), not Olist. Truck: warehouse W to hub H is 100 km,
1 BRL per km, 50 BRL per trip, so one trip costs 250. Two vans of 30 kg, 1 BRL per km. Holding
1 BRL per kg per day. Hub limit 60 kg, empty at the start. Km: H-A 4, H-B 6, H-C 5, A-B 3,
A-C 6, B-C 4.

| Point | Demand day 0 | Demand day 1 | Storage limit | Start stock |
| --- | ---: | ---: | ---: | ---: |
| A | 10 | 10 | 20 | 0 |
| B | 20 | 0 | 20 | 0 |
| C | 5 | 15 | 20 | 5 |

| Method | Day 0 routes, km (kg) | Day 1 routes, km (kg) | A gets | Truck | Van | Holding | Total |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Greedy | H-A-H 8 (20), H-B-H 12 (20) | H-C-H 10 (15) | 20, 0 | 250 | 30 | 25 | 305 |
| Gurobi MIP | H-A-B-H 13 (30) | H-A-C-H 15 (25) | 10, 10 | 250 | 28 | 25 | 303 |
| ALNS, seed 1 | H-B-A-H 13 (30) | H-C-A-H 15 (25) | 10, 10 | 250 | 28 | 25 | 303 |
| Matheuristic, seed 1 | H-B-A-H 13 (30) | H-C-A-H 15 (25) | 10, 10 | 250 | 28 | 25 | 303 |

Every plan has one truck trip on day 0 with 55 kg. B gets 20 kg on day 0 and C 15 kg on day 1.
The checker says "feasible" for all four, and Gurobi proves 303 optimal. The script also tries
all 102 feasible plans: the three cheapest cost 303, 304 and 305. The ALNS and the matheuristic
start at 305 and reach 303 within 2,000 iterations. Their routes are the MIP routes driven the
other way round, so the km are the same.

### Why the optimum beats the greedy

The greedy fills A to 20 kg on day 0. Then A and B need 40 kg,
more than one 30 kg van, so day 0 needs two vans. The optimum gives A only 10 kg on day 0, so A
and B fill exactly one van. A gets its other 10 kg on day 1, when the van passes A on the way to
C anyway. Van cost falls from 30 to 28. Holding is 25 in both (greedy: 15 kg at H and 10 kg at A
overnight; optimum: 25 kg at H). A second truck trip would cost 250, so no plan uses one.

### The LP at work

In the matheuristic run, J1 failed 32 times and the LP fixed none of them. To
show a rescue, the script cuts the vans to 22 kg. It fixes one plan: day 0 routes [A] and
[B, C], day 1 route [A, C]. J1 puts 25 kg on the day 1 route and rejects the plan. The LP moves
kg earlier: A gets 11 and 9, C gets 2 and 13. Every route carries at most 22 kg, holding is 25,
and the checker says "feasible".

## 10 likely interview questions

### 1. Why ALNS?

It was built for vehicle routing (Ropke and Pisinger 2006) and has been used
for inventory routing (Coelho, Cordeau and Laporte 2012). Each new idea is one more destroy or
repair rule, and the weights learn which rules work. On all 12 small instances it found the
proven optimum in all 10 seeds, 10 seconds per run (`results/tables/gap_to_optimum.md`).

### 2. Why a matheuristic?

Once the routes are fixed, the kg are a linear problem, so a solver
can set them exactly. J1 is fast but gives up when a van is overloaded. The LP can then move part
of a delivery to an earlier visit and save the plan. I chose a 650 kg van (Fiat Fiorino), so
van capacity can bind at all. With the larger van I tried first, the LP was never called.

### 3. Why did the matheuristic not win?

The LP helps only when J1 fails. That never happened
on the small and medium sets (0 LP calls). On the large set J1 failed 25 to 628 times per run
(`results/runs_compare.csv`). The mean number of rescues per run is 180.5 to 220.8, by instance
(`results/tables/summary.md`). The LP costs time: on `large_k5_n200_T7_s1` the matheuristic ran
4,575.9 iterations on average against 5,213.0 for the ALNS. Wilcoxon finds no significant
difference on any medium or large instance (p from 0.1934 to 0.6953, `results/tables/wilcoxon.md`).

### 4. How do you make sure every solution is feasible?

No penalty costs are used. The repair adds visits and trips until nothing runs out, and a plan
that overloads a van is rejected. One checker (`src/irp2e/evaluation.py`) tests 13 rules on the
final solution of every run. A 14th rule compares a reported cost with the recomputed cost.
`scripts/analyze.py` checks every saved solution again before it writes any table.

### 5. How do you know the MIP is right?

On the tiny example it gives 303. `scripts/worked_example.py` also tries all 102 feasible plans
of this example, with the quantity LP for each, and finds the same 303. The checker recomputes the cost of every Gurobi solution and compares it with Gurobi's objective. On all 12
small instances the ALNS, a separate code, reaches exactly the MIP optimum. Tests pin these
values.

### 6. Why the Wilcoxon signed-rank test?

The runs are paired by seed, and 10 runs are too few
to assume a normal distribution. The test uses only the signs and ranks of the differences
(Wilcoxon 1945). The greedy has one run, so each method is tested on its 10 differences to the
greedy. With 10 pairs the smallest two-sided p-value is 0.0020, so every "beats greedy" row
shows 0.0020: all 10 runs were cheaper.

### 7. What does the sensitivity study show?

Holding cost is only the cost of capital, so it is
tiny. In the mean ALNS plan of `medium_k3_n80_T7_s1` it is 98.58 of 36,848.41 BRL
(`results/tables/sensitivity.md`). With holding cost times 1,000, the ALNS uses 13.00 truck
trips instead of 6.00 and makes more visits. The greedy plan never changes, so at times 10,000
it costs 1,393,911.78 BRL against 216,300.19 BRL for the ALNS.

### 8. How was the data built?

Orders are the public Olist data (CC BY-NC-SA 4.0), Sao Paulo
state, delivered orders only. One year from 2017-08-07 is folded onto one week by weekday. Each
5-digit zip prefix is one pickup point. The warehouse is the prefix with the most sellers
(Ibitinga). Each hub is the busiest prefix of one of the five cities with the most delivered kg.
Road km come from OSRM on OpenStreetMap (ODbL), then the shortest path through the instance
nodes, so every km obeys the triangle rule.

### 9. What are the limits of the model?

Demand is known in advance. A truck serves one hub per
trip, and driving time is ignored. The van cost per km reuses the truck rate, so it is an upper
bound. Costs are from 2026, orders from 2017 and 2018. Medium and large instances have no proven
optimum, only a best known cost.

### 10. What would you do next?

First, truck tours that visit several hubs, because some hubs
are close to each other. Second, a switch-off test (in the literature: ablation) that runs the
ALNS without each operator, because weights alone do not prove an operator's value. Third, lower
bounds for the medium instances, to measure the real gap.
