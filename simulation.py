import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Simulation settings
T = 30  # number of years
n = 100  # number of Monte Carlo iterations
R_star = 0.8  # Resilience threshold
initial_absorptive_capacity = 0.3
initial_governance = 0.358
initial_population = 477088526
initial_gni_per_capita = 1342.05
carrying_capacity = 60e6

# Constants
beta = {
    "healthcare": 0.9,
    "agriculture": 1.707,
    "infrastructure": 1.1,
    "technology": 0.8,
    "transport": 1.0,
    "education": 0.0065
}
delta = {
    "healthcare": 0.0064,
    "agriculture": 0.7799,
    "infrastructure": 0.0313,
    "technology": 0.0345,
    "transport": 0.0039,
    "education": 0.0077
}
E = {
    "healthcare": 0.0136,
    "agriculture": 0.0339,
    "infrastructure": 0.0083,
    "technology": 0.0397,
    "transport": 0.01,
    "education": 0.0612
}
theta = 1.18
psi = 3.421e-6
lambda_param = 0.5

# Initial values
initial_resilience = {
    "healthcare": 0.8218,
    "agriculture": 0.5192,
    "infrastructure": 0.24,
    "technology": 0.0951,
    "transport": 0.1198,
    "education": 0.538
}

sectors = list(initial_resilience.keys())

# Storage for simulation results
results = []

# Monte Carlo simulation (n=1)
years = []
R_means = []
for iteration in range(n):
    R = initial_resilience.copy()
    gamma = initial_governance
    alpha = initial_absorptive_capacity
    GNI = initial_population * initial_gni_per_capita
    population = initial_population
    A_total = 100e6  # constant annual aid

    for t in range(T):
        np.random.seed(42)
        a_i = np.random.dirichlet(np.ones(len(sectors)))  # random allocation
        A_i = {}
        A_i_effective = {}
        R_new = {}
        R_average = np.mean([R[sector] for sector in sectors])

        # Sector updates
        for idx, sector in enumerate(sectors):
            if R[sector] >= R_star:
                a_i[idx] = 0

        a_i = a_i / np.sum(a_i)  # normalize

        for idx, sector in enumerate(sectors):
            A_i[sector] = a_i[idx] * A_total
            A_i_effective[sector] = alpha * A_i[sector]

            # print(R[sector])
            R_change = np.log(1 + (1 - R[sector]) * (beta[sector] / 100) * (A_i_effective[sector] / 1e6))

            # print(mult)
            # print("")

            # diminishing_returns = (1 - R[sector]) * (1 + beta[sector] / 100)
            # log_term = math.log10(1 + diminishing_returns * (A_i_effective[sector] / 1e6))
            domestic_growth = (delta[sector] / 100) * (1 + gamma / 100)
            R_new[sector] = min(1, R[sector] + R_change + domestic_growth)

            # print(R_new[sector])

        # Governance update
        total_effective_aid = sum(A_i_effective.values())
        gamma_new = min(1, gamma + psi * (total_effective_aid / 1e6))

        # print(gamma, gamma_new)

        # Absorptive capacity update
        alpha = min(1, initial_absorptive_capacity + lambda_param * gamma_new)

        # print(alpha)

        # GNI updates
        delta_R = {sector: R_new[sector] - R[sector] for sector in sectors}
        gni_growth_sector = sum((E[sector]) * delta_R[sector] for sector in sectors)

        phi = sum((E[sector]) * (delta[sector] / 100) * (1 + gamma / 100) for sector in sectors)

        gni_growth_governance = (theta / 100) * (gamma_new - gamma)
        gni_growth_rate = phi + gni_growth_sector + gni_growth_governance
        GNI *= (1 + gni_growth_rate)
        #
        # print(GNI)
    #
        # Population update (logistic growth)
        growth_rate = 0.03113191
        population = population + population * growth_rate * (1 - population / carrying_capacity)

        # Update GNI per capita
        GNI_pc = GNI / population

        # print(GNI_pc)
    #
        # Store results

        results.append({
            "year": t+2015,
            "population": population,
            "GNI": GNI,
            "GNI_pc": GNI_pc,
            "governance": gamma,
            "absorptive_capacity": alpha,
            **{f"R_{sector}": R[sector] for sector in sectors},
            "R_average": np.mean([R[sector] for sector in sectors])
        })
    #
        # Carry over for next year
        R = R_new.copy()
        gamma = gamma_new

        if R_average >= R_star:
            years.append(t)
            R_means.append(R_average)
            # print(R)
            break


def optimize_fixed_budget(aid_budget, n=100, T=30):
    best_years = T
    best_allocation = None
    best_path = []

    for _ in range(n):
        allocation = np.random.dirichlet(np.ones(len(sectors)))
        path = simulate_path(aid_budget, allocation, T)
        if path[-1]["R_average"] >= R_star and len(path) < best_years:
            best_years = len(path)
            best_allocation = allocation
            best_path = path

    return best_allocation, best_years, best_path

print(years)
# Output to DataFrame
df_results = pd.DataFrame(results)
print(df_results)

real_results = pd.read_csv('kenya_sector_gni_data.csv', skiprows=range(1, 17))

y1 = list(df_results.loc[:,"GNI_pc"])
y2 = real_results.loc[:,"gni pc ppp"]

print(y1)
print(years)
print(min(years))

def simulate_until_graduation(aid_budget, allocation, T=30):
    R = initial_resilience.copy()
    gamma = initial_governance
    alpha = initial_absorptive_capacity
    GNI = initial_population * initial_gni_per_capita
    population = initial_population

    for t in range(T):
        A_i = {}
        A_i_effective = {}
        R_new = {}

        # Sector updates
        for idx, sector in enumerate(sectors):
            if R[sector] >= R_star:
                allocation[idx] = 0
        allocation = np.array(allocation)
        allocation = allocation / allocation.sum()

        for idx, sector in enumerate(sectors):
            A_i[sector] = allocation[idx] * aid_budget
            A_i_effective[sector] = alpha * A_i[sector]
            R_change = np.log(1 + (1 - R[sector]) * (beta[sector] / 100) * (A_i_effective[sector] / 1e6))
            domestic_growth = (delta[sector] / 100) * (1 + gamma / 100)
            R_new[sector] = min(1, R[sector] + R_change + domestic_growth)

        total_effective_aid = sum(A_i_effective.values())
        gamma_new = min(1, gamma + psi * (total_effective_aid / 1e6))
        alpha = min(1, initial_absorptive_capacity + lambda_param * gamma_new)

        delta_R = {sector: R_new[sector] - R[sector] for sector in sectors}
        gni_growth_sector = sum((E[sector]) * delta_R[sector] for sector in sectors)
        phi = sum((E[sector]) * (delta[sector] / 100) * (1 + gamma / 100) for sector in sectors)
        gni_growth_governance = (theta / 100) * (gamma_new - gamma)
        gni_growth_rate = phi + gni_growth_sector + gni_growth_governance
        GNI *= (1 + gni_growth_rate)

        growth_rate = 0.03113191
        population = population + population * growth_rate * (1 - population / carrying_capacity)
        GNI_pc = GNI / population

        R = R_new.copy()
        gamma = gamma_new

        if np.mean(list(R.values())) >= R_star:
            return t + 1

    return T + 1  # return high value if not graduated


def find_min_aid_to_graduate(target_years, allocation=None, T=30, precision=1e5, max_iters=20):
    if allocation is None:
        allocation = np.random.dirichlet(np.ones(len(sectors)))

    low = 10e6  # lower bound (unrealistically low)
    high = 2e9  # upper bound (unrealistically high)
    best_aid = high

    for _ in range(max_iters):
        mid = (low + high) / 2
        years = simulate_until_graduation(mid, allocation, T)

        if years <= target_years:
            best_aid = mid
            high = mid - precision
        else:
            low = mid + precision

        if high - low < precision:
            break

    return best_aid, allocation


plt.plot(list(2015 + i for i in range(len(y2))), y1[:7], 'r', label="Optimised results")
plt.plot(list(2015 + i for i in range(len(y2))), y2, 'g', label="Real results")
plt.title("Optimised aid allocation results for GNI vs real results")
plt.ylabel("Resilience scores")
plt.xlabel("Time, years")
plt.legend()
plt.show()

