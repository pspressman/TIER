import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate

# Number of runs
n_runs = 2000
iterations = 20

# Arrays to store results across runs
all_work_a = np.zeros((n_runs, iterations))
all_work_c = np.zeros((n_runs, iterations))
all_work_g = np.zeros((n_runs, iterations))
all_global_thermodynamic_entropy = np.zeros((n_runs, iterations))

all_informational_entropy_a = np.zeros((n_runs, iterations))
all_informational_entropy_c = np.zeros((n_runs, iterations))
all_informational_entropy_g = np.zeros((n_runs, iterations))
all_global_informational_entropy = np.zeros((n_runs, iterations))

all_lyapunov_level1 = np.zeros((n_runs, iterations))
all_lyapunov_level2 = np.zeros((n_runs, iterations))
all_lyapunov_level3 = np.zeros((n_runs, iterations))

def activation(x):
    return np.tanh(x)

def generate_input():
    base = np.random.uniform(0.4, 0.6)
    return base + np.random.uniform(-0.05, 0.05)

def calculate_entropy(prob):
    return -prob * np.log(prob) - (1 - prob) * np.log(1 - prob)

def lyapunov(w, target):
    return 0.5 * np.sum((w - target)**2)

def calculate_dynamics_metrics(data_array):
    # Starting points (mean and SD across runs)
    start_mean = np.mean(data_array[:, 0])
    start_sd = np.std(data_array[:, 0])
    
    # Ending points (mean and SD across runs)
    end_mean = np.mean(data_array[:, -1])
    end_sd = np.std(data_array[:, -1])
    
    # Calculate AUC for each run
    aucs = np.array([integrate.trapz(run_data) for run_data in data_array])
    auc_mean = np.mean(aucs)
    auc_sd = np.std(aucs)
    
    return {
        'start': (start_mean, start_sd),
        'end': (end_mean, end_sd),
        'auc': (auc_mean, auc_sd)
    }

# Run simulation multiple times
for run in range(n_runs):
    # Initialize weights for all modules
    w_a1, w_a2 = np.random.uniform(0.1, 0.9, 2)
    w_b1, w_b2 = np.random.uniform(0.1, 0.9, 2)
    w_d1, w_d2 = np.random.uniform(0.1, 0.9, 2)
    w_e1, w_e2 = np.random.uniform(0.1, 0.9, 2)
    w_c1, w_c2 = np.random.uniform(0.1, 0.9, 2)
    w_f1, w_f2 = np.random.uniform(0.1, 0.9, 2)
    w_g1, w_g2 = np.random.uniform(0.1, 0.9, 2)

    work_a, work_d, work_c, work_f, work_g = [], [], [], [], []
    informational_entropy_a, informational_entropy_c, informational_entropy_d = [], [], []
    informational_entropy_f, informational_entropy_g = [], []
    global_thermodynamic_entropy = []
    global_informational_entropy = []
    lyapunov_level1, lyapunov_level2, lyapunov_level3 = [], [], []

    # Your existing simulation loop
    for step in range(iterations):
        # Sensory inputs for A and B
        x_a1 = generate_input()
        x_a2 = generate_input()
        x_b1 = generate_input()
        x_b2 = generate_input()
        integrated_a = (w_a1 * x_a1) + (w_a2 * x_a2)
        integrated_b = (w_b1 * x_b1) + (w_b2 * x_b2)
        out_a = activation(integrated_a)
        out_b = activation(integrated_b)
        delta_w_a = abs(w_a1 - x_a1) + abs(w_a2 - x_a2)
        delta_w_b = abs(w_b1 - x_b1) + abs(w_b2 - x_b2)
        work_a.append(delta_w_a)
        w_a1 += 0.1 * (x_a1 - w_a1)
        w_a2 += 0.1 * (x_a2 - w_a2)
        w_b1 += 0.1 * (x_b1 - w_b1)
        w_b2 += 0.1 * (x_b2 - w_b2)
        prob_a = (out_a + 1) / 2
        informational_entropy_a.append(calculate_entropy(prob_a))
        
        # For D and E
        x_d1 = generate_input()
        x_d2 = generate_input()
        x_e1 = generate_input()
        x_e2 = generate_input()
        integrated_d = (w_d1 * x_d1) + (w_d2 * x_d2)
        integrated_e = (w_e1 * x_e1) + (w_e2 * x_e2)
        out_d = activation(integrated_d)
        out_e = activation(integrated_e)
        delta_w_d = abs(w_d1 - x_d1) + abs(w_d2 - x_d2)
        delta_w_e = abs(w_e1 - x_e1) + abs(w_e2 - x_e2)
        work_d.append(delta_w_d)
        w_d1 += 0.1 * (x_d1 - w_d1)
        w_d2 += 0.1 * (x_d2 - w_d2)
        w_e1 += 0.1 * (x_e1 - w_e1)
        w_e2 += 0.1 * (x_e2 - w_e2)
        prob_d = (out_d + 1) / 2
        informational_entropy_d.append(calculate_entropy(prob_d))

        # Integration at node C
        integrated_c = w_c1 * out_a + w_c2 * out_b
        out_c = activation(integrated_c)
        delta_w_c = abs(w_c1 - out_a) + abs(w_c2 - out_b)
        work_c.append(delta_w_c)
        w_c1 += 0.1 * (out_a - w_c1)
        w_c2 += 0.1 * (out_b - w_c2)
        prob_c = (out_c + 1) / 2
        informational_entropy_c.append(calculate_entropy(prob_c))

        # F integration
        integrated_f = w_f1 * out_d + w_f2 * out_e
        out_f = activation(integrated_f)
        delta_w_f = abs(w_f1 - out_d) + abs(w_f2 - out_e)
        work_f.append(delta_w_f)
        w_f1 += 0.1 * (out_d - w_f1)
        w_f2 += 0.1 * (out_e - w_f2)
        prob_f = (out_f + 1) / 2
        informational_entropy_f.append(calculate_entropy(prob_f))

        # Top node G
        integrated_g = w_g1 * out_c + w_g2 * out_f
        out_g = activation(integrated_g)
        delta_w_g = abs(w_g1 - out_c) + abs(w_g2 - out_c)
        work_g.append(delta_w_g)
        w_g1 += 0.1 * (out_c - w_g1)
        w_g2 += 0.1 * (out_f - w_g2)
        prob_g = (out_g + 1) / 2
        informational_entropy_g.append(calculate_entropy(prob_g))

        # Calculate Lyapunov values
        V1 = lyapunov(np.array([w_a1, w_a2]), np.array([x_a1, x_a2]))
        lyapunov_level1.append(V1)
        
        V2 = lyapunov(np.array([w_c1, w_c2]), np.array([out_a, out_b]))
        lyapunov_level2.append(V2)
        
        V3 = lyapunov(np.array([w_g1, w_g2]), np.array([out_c, out_c]))
        lyapunov_level3.append(V3)

        # Global measures
        global_thermodynamic_entropy.append(delta_w_a + delta_w_c + delta_w_g)
        global_entropy = informational_entropy_a[-1] + informational_entropy_c[-1] + informational_entropy_g[-1]
        global_informational_entropy.append(global_entropy)

    # Store results for this run
    all_work_a[run] = work_a
    all_work_c[run] = work_c
    all_work_g[run] = work_g
    all_global_thermodynamic_entropy[run] = global_thermodynamic_entropy
    all_informational_entropy_a[run] = informational_entropy_a
    all_informational_entropy_c[run] = informational_entropy_c
    all_informational_entropy_g[run] = informational_entropy_g
    all_global_informational_entropy[run] = global_informational_entropy
    all_lyapunov_level1[run] = lyapunov_level1
    all_lyapunov_level2[run] = lyapunov_level2
    all_lyapunov_level3[run] = lyapunov_level3

# Calculate means and confidence intervals
confidence_level = 0.95
time_steps = range(iterations)

def calculate_confidence_interval(data):
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
    return mean, ci[0], ci[1]

# Calculate statistics for all measures
mean_work_a, ci_low_work_a, ci_high_work_a = calculate_confidence_interval(all_work_a)
mean_work_c, ci_low_work_c, ci_high_work_c = calculate_confidence_interval(all_work_c)
mean_work_g, ci_low_work_g, ci_high_work_g = calculate_confidence_interval(all_work_g)
mean_global_therm, ci_low_global_therm, ci_high_global_therm = calculate_confidence_interval(all_global_thermodynamic_entropy)

mean_info_a, ci_low_info_a, ci_high_info_a = calculate_confidence_interval(all_informational_entropy_a)
mean_info_c, ci_low_info_c, ci_high_info_c = calculate_confidence_interval(all_informational_entropy_c)
mean_info_g, ci_low_info_g, ci_high_info_g = calculate_confidence_interval(all_informational_entropy_g)
mean_global_info, ci_low_global_info, ci_high_global_info = calculate_confidence_interval(all_global_informational_entropy)

mean_lyap1, ci_low_lyap1, ci_high_lyap1 = calculate_confidence_interval(all_lyapunov_level1)
mean_lyap2, ci_low_lyap2, ci_high_lyap2 = calculate_confidence_interval(all_lyapunov_level2)
mean_lyap3, ci_low_lyap3, ci_high_lyap3 = calculate_confidence_interval(all_lyapunov_level3)

# Plotting with confidence intervals
plt.figure(figsize=(14, 10))

# Thermodynamic Entropy
plt.subplot(3, 1, 1)
plt.plot(time_steps, mean_work_a, label='Level 1', color='blue')
plt.fill_between(time_steps, ci_low_work_a, ci_high_work_a, color='blue', alpha=0.2)
plt.plot(time_steps, mean_work_c, label='Level 2', color='orange')
plt.fill_between(time_steps, ci_low_work_c, ci_high_work_c, color='orange', alpha=0.2)
plt.plot(time_steps, mean_work_g, label='Level 3', color='green')
plt.fill_between(time_steps, ci_low_work_g, ci_high_work_g, color='green', alpha=0.2)
plt.plot(time_steps, mean_global_therm, label='Global', linestyle='--', color='red')
plt.fill_between(time_steps, ci_low_global_therm, ci_high_global_therm, color='red', alpha=0.2)
plt.title('Thermodynamic Entropy Over Time (with 95% CI)')
plt.xlabel('Time Steps')
plt.ylabel('Work')
plt.legend()

# Informational Entropy
plt.subplot(3, 1, 2)
plt.plot(time_steps, mean_info_a, label='Level 1', color='blue')
plt.fill_between(time_steps, ci_low_info_a, ci_high_info_a, color='blue', alpha=0.2)
plt.plot(time_steps, mean_info_c, label='Level 2', color='orange')
plt.fill_between(time_steps, ci_low_info_c, ci_high_info_c, color='orange', alpha=0.2)
plt.plot(time_steps, mean_info_g, label='Level 3', color='green')
plt.fill_between(time_steps, ci_low_info_g, ci_high_info_g, color='green', alpha=0.2)
plt.title('Informational Entropy Over Time (with 95% CI)')
plt.xlabel('Time Steps')
plt.ylabel('Entropy')
plt.legend()

# Global Informational Entropy
plt.subplot(3, 1, 3)
plt.plot(time_steps, mean_global_info, label='Global Information Entropy', color='purple')
plt.fill_between(time_steps, ci_low_global_info, ci_high_global_info, color='purple', alpha=0.2)
plt.title('Global Informational Entropy Over Time (with 95% CI)')
plt.xlabel('Time Steps')
plt.ylabel('Entropy')
plt.legend()

plt.tight_layout()
plt.show()

# Lyapunov plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, mean_lyap1, label='Level 1', color='blue')
plt.fill_between(time_steps, ci_low_lyap1, ci_high_lyap1, color='blue', alpha=0.2)
plt.plot(time_steps, mean_lyap2, label='Level 2', color='orange')
plt.fill_between(time_steps, ci_low_lyap2, ci_high_lyap2, color='orange', alpha=0.2)
plt.plot(time_steps, mean_lyap3, label='Level 3', color='green')
plt.fill_between(time_steps, ci_low_lyap3, ci_high_lyap3, color='green', alpha=0.2)
plt.title('Lyapunov Function Values Over Time (with 95% CI)')
plt.xlabel('Time Steps')
plt.ylabel('V(w)')
plt.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics across all runs:")
print("\nMean Final Values (± 95% CI):")
print(f"Level 1 Lyapunov: {mean_lyap1[-1]:.4f} ± {(ci_high_lyap1[-1] - ci_low_lyap1[-1])/2:.4f}")
print(f"Level 2 Lyapunov: {mean_lyap2[-1]:.4f} ± {(ci_high_lyap2[-1] - ci_low_lyap2[-1])/2:.4f}")
print(f"Level 3 Lyapunov: {mean_lyap3[-1]:.4f} ± {(ci_high_lyap3[-1] - ci_low_lyap3[-1])/2:.4f}")

# Add statistical tests to compare levels


# Lyapunov Analysis
print("\nLyapunov Dynamics Analysis:")
for level, data in [
    ("Level 1", all_lyapunov_level1),
    ("Level 2", all_lyapunov_level2),
    ("Level 3", all_lyapunov_level3)
]:
    metrics = calculate_dynamics_metrics(data)
    print(f"\n{level}:")
    print(f"Start: {metrics['start'][0]:.4f} ± {metrics['start'][1]:.4f}")
    print(f"End: {metrics['end'][0]:.4f} ± {metrics['end'][1]:.4f}")
    print(f"AUC: {metrics['auc'][0]:.4f} ± {metrics['auc'][1]:.4f}")

# Thermodynamic Entropy Analysis
print("\nThermodynamic Entropy Dynamics Analysis:")
for level, data in [
    ("Level 1", all_work_a),
    ("Level 2", all_work_c),
    ("Level 3", all_work_g)
]:
    metrics = calculate_dynamics_metrics(data)
    print(f"\n{level}:")
    print(f"Start: {metrics['start'][0]:.4f} ± {metrics['start'][1]:.4f}")
    print(f"End: {metrics['end'][0]:.4f} ± {metrics['end'][1]:.4f}")
    print(f"AUC: {metrics['auc'][0]:.4f} ± {metrics['auc'][1]:.4f}")

# Informational Entropy Analysis
print("\nInformational Entropy Dynamics Analysis:")
for level, data in [
    ("Level 1", all_informational_entropy_a),
    ("Level 2", all_informational_entropy_c),
    ("Level 3", all_informational_entropy_g)
]:
    metrics = calculate_dynamics_metrics(data)
    print(f"\n{level}:")
    print(f"Start: {metrics['start'][0]:.4f} ± {metrics['start'][1]:.4f}")
    print(f"End: {metrics['end'][0]:.4f} ± {metrics['end'][1]:.4f}")
    print(f"AUC: {metrics['auc'][0]:.4f} ± {metrics['auc'][1]:.4f}")

