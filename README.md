# Astar Island — NM i AI 2026

A prediction agent for the **Astar Island** challenge in the [Norwegian AI Championship 2026](https://app.ainm.no).

## What is it?

A Norse settlement simulation runs for 50 years on a 40×40 grid. Settlements grow, raid each other, trade, and collapse — driven by stochastic mechanics. The goal is to predict the **final terrain state** of the map before seeing it.

Each round you get:
- The **initial map state** for 5 seeds (terrain + starting settlements)
- A budget of **50 simulation queries** to observe random viewport snapshots of simulation runs
- A ~2h45m window to submit your predictions

## The Task

Submit a **probability tensor** of shape `H × W × 6` for each of the 5 seeds, where each cell contains a probability distribution over 6 terrain classes:

| Index | Class |
|-------|-------|
| 0 | Empty (Ocean / Plains / Empty) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |

## Scoring

Scored using **entropy-weighted KL divergence** against a ground truth computed from hundreds of hidden simulation runs:

```
score = max(0, min(100, 100 × exp(-3 × weighted_KL)))
```

- **100** = perfect match
- Static cells (ocean, mountains) are excluded — only dynamic cells count
- ⚠️ Never assign `0.0` to any class — it sends KL divergence to infinity

Always apply a probability floor:
```python
pred = np.maximum(pred, 0.01)
pred = pred / pred.sum(axis=-1, keepdims=True)
```

## Approach

1. **Fetch** the initial map state to identify static vs. dynamic cells
2. **Sample** simulation viewports to build observed frequency distributions
3. **Build** a prediction tensor using observed frequencies + terrain-based priors
4. **Submit** all 5 seeds before the round closes (a uniform prediction beats a missing one)

## API

Base URL: `https://api.ainm.no/astar-island`

| Endpoint | Description |
|----------|-------------|
| `GET /rounds` | List rounds + status |
| `GET /rounds/{id}` | Initial map state for all seeds |
| `GET /budget` | Remaining query budget |
| `POST /simulate` | Query a viewport snapshot (costs 1 query) |
| `POST /submit` | Submit prediction tensor for a seed |

Auth: `Authorization: Bearer <token>`

## Project Structure

```
.
├── main.py          # Entry point — runs full pipeline for active round
├── simulate.py      # Viewport sampling + observation collection
├── predict.py       # Build prediction tensor from observations
├── submit.py        # Submit predictions to the API
└── utils.py         # Helpers (auth, terrain mapping, probability floor)
```

## Quickstart

```bash
pip install numpy requests
export AINM_TOKEN="your-jwt-token"
python main.py
```
