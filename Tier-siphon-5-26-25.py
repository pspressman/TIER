import numpy as np
import matplotlib.pyplot as plt

# Define physiological parameters
F_max = 5.0  # Physiological force limit

def run_tier_simulation(time_steps=2000, dt=0.01):
    """
    TIER model using only defined variables: cells, F, D, W, FFE, N_failure
    No arbitrary additions.
    """
    
    # Initialize cell populations
    cortical_cells = np.zeros(time_steps)
    support_cells = np.zeros(time_steps)
    cortical_cells[0] = 1000
    support_cells[0] = 300
    
    # System resilience parameters (scaled for visible effects)
    FFE_cortex = 6.0      # Higher resilience
    FFE_support = 2.0     # Lower resilience (3:1 ratio maintained)
    
    # Capacity per cell
    D_per_cell = 1.0
    
    # Constant cognitive work demand
    required_cognitive_work = 3000
    
    # Initialize tracking arrays
    W_cortex = np.zeros(time_steps)
    W_support = np.zeros(time_steps)
    F_cortex = np.zeros(time_steps)
    F_support = np.zeros(time_steps)
    work_per_cortical_cell = np.zeros(time_steps)
    work_per_support_cell = np.zeros(time_steps)
    accumulated_intensity_cortex = np.zeros(time_steps)
    accumulated_intensity_support = np.zeros(time_steps)
    N_failure_cortex = np.zeros(time_steps)
    N_failure_support = np.zeros(time_steps)
    
    # Simulation loop
    for t in range(1, time_steps):
        current_cortical = max(1, cortical_cells[t-1])
        current_support = max(1, support_cells[t-1])
        
        # Total capacity available
        D_cortex_total = current_cortical * D_per_cell
        D_support_total = current_support * D_per_cell
        
        # Support systems provide boost proportional to cortical cell loss
        # More cortical cells lost = more boost needed per remaining cortical cell
        cortical_cells_lost = 1000 - current_cortical
        support_boost_per_cortical_cell = cortical_cells_lost / 1000  # 0 to 1 scale
        total_support_boost = support_boost_per_cortical_cell * current_cortical * 2.0
        
        # Support work = baseline support + compensation boost
        baseline_support_work = 500  # Support always working
        W_support_needed = baseline_support_work + total_support_boost
        
        # Calculate support force needed
        F_support[t] = min(W_support_needed / D_support_total, F_max) if D_support_total > 0 else 0
        W_support[t] = F_support[t] * D_support_total
        
        # Cortical work = remaining work after support contribution
        W_cortex_needed = required_cognitive_work - W_support[t]
        
        # Calculate cortical force needed
        F_cortex[t] = min(W_cortex_needed / D_cortex_total, F_max) if D_cortex_total > 0 else 0
        W_cortex[t] = F_cortex[t] * D_cortex_total
        
        # Calculate work per individual cell (INTENSITY)
        work_per_cortical_cell[t] = W_cortex[t] / current_cortical if current_cortical > 0 else 0
        work_per_support_cell[t] = W_support[t] / current_support if current_support > 0 else 0
        
        # Accumulate work intensity over time
        accumulated_intensity_cortex[t] = accumulated_intensity_cortex[t-1] + work_per_cortical_cell[t] * dt
        accumulated_intensity_support[t] = accumulated_intensity_support[t-1] + work_per_support_cell[t] * dt
        
        # System failure rates driven by accumulated intensity
        N_failure_cortex[t] = A_cortex * np.exp(accumulated_intensity_cortex[t] / FFE_cortex)
        N_failure_support[t] = A_support * np.exp(accumulated_intensity_support[t] / FFE_support)
        
        # Apply cell death
        cells_lost_cortex = N_failure_cortex[t] * dt
        cells_lost_support = N_failure_support[t] * dt
        
        # Update populations
        cortical_cells[t] = max(0, cortical_cells[t-1] - cells_lost_cortex)
        support_cells[t] = max(0, support_cells[t-1] - cells_lost_support)
    
    return {
        'time': np.arange(time_steps) * dt,
        'cortical_cells': cortical_cells,
        'support_cells': support_cells,
        'W_cortex': W_cortex,
        'W_support': W_support,
        'F_cortex': F_cortex,
        'F_support': F_support,
        'work_per_cortical_cell': work_per_cortical_cell,
        'work_per_support_cell': work_per_support_cell,
        'accumulated_intensity_cortex': accumulated_intensity_cortex,
        'accumulated_intensity_support': accumulated_intensity_support,
        'N_failure_cortex': N_failure_cortex,
        'N_failure_support': N_failure_support,
        'required_cognitive_work': required_cognitive_work,
        'total_work': W_cortex + W_support
    }

# Run simulation
print("Running TIER simulation...")
results = run_tier_simulation()

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TIER Model: Siphon Effect and Work Intensity', fontsize=16, fontweight='bold')

# 1. Survival Curves - Absolute Numbers
axes[0,0].plot(results['time'], results['cortical_cells'], 'b-', label='Cortical cells', linewidth=2.5)
axes[0,0].plot(results['time'], results['support_cells'], 'r-', label='Support cells', linewidth=2.5)
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Cell count')
axes[0,0].set_title('A. Absolute Cell Survival')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Survival Curves - Relative (Percentage)
axes[0,1].plot(results['time'], results['cortical_cells']/1000 * 100, 'b-', 
               label='Cortical (% of 1000)', linewidth=2.5)
axes[0,1].plot(results['time'], results['support_cells']/300 * 100, 'r-', 
               label='Support (% of 300)', linewidth=2.5)
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Percent surviving')
axes[0,1].set_title('B. Relative Cell Survival')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. System Failure Rates
axes[0,2].plot(results['time'], results['N_failure_cortex'], 'b-', 
               label='Cortical failure rate', linewidth=2.5)
axes[0,2].plot(results['time'], results['N_failure_support'], 'r-', 
               label='Support failure rate', linewidth=2.5)
axes[0,2].set_xlabel('Time')
axes[0,2].set_ylabel('Failure rate (cells/time)')
axes[0,2].set_title('C. System Failure Rates')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Work Intensity Per Cell
axes[1,0].plot(results['time'], results['work_per_cortical_cell'], 'b-', 
               label='Cortical work/cell', linewidth=2.5)
axes[1,0].plot(results['time'], results['work_per_support_cell'], 'r-', 
               label='Support work/cell', linewidth=2.5)
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Work per cell')
axes[1,0].set_title('D. Individual Cell Work Intensity')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5. Force Utilization Over Time
axes[1,1].plot(results['time'], results['F_cortex'], 'b-', label='Cortical force', linewidth=2.5)
axes[1,1].plot(results['time'], results['F_support'], 'r-', label='Support force', linewidth=2.5)
axes[1,1].axhline(y=F_max, color='k', linestyle='--', alpha=0.5, label='F_max (both systems)')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Force')
axes[1,1].set_title('E. Force Utilization')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# 6. Accumulated Work Intensity (Entropy Drivers)
axes[1,2].plot(results['time'], results['accumulated_intensity_cortex'], 'b-', 
               label='Accumulated cortical intensity', linewidth=2.5)
axes[1,2].plot(results['time'], results['accumulated_intensity_support'], 'r-', 
               label='Accumulated support intensity', linewidth=2.5)
axes[1,2].set_xlabel('Time')
axes[1,2].set_ylabel('Accumulated work intensity')
axes[1,2].set_title('F. Entropy Accumulation')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary Statistics
print("\n" + "="*60)
print("TIER MODEL RESULTS")
print("="*60)

# Find when each system drops to key thresholds
def find_threshold_time(population, threshold_fraction, initial_count):
    threshold = initial_count * threshold_fraction
    crossing = np.where(population <= threshold)[0]
    return crossing[0] * 0.01 if len(crossing) > 0 else None

support_50_time = find_threshold_time(results['support_cells'], 0.5, 300)
cortical_50_time = find_threshold_time(results['cortical_cells'], 0.5, 1000)
support_10_time = find_threshold_time(results['support_cells'], 0.1, 300)
cortical_10_time = find_threshold_time(results['cortical_cells'], 0.1, 1000)

print(f"Population decline thresholds:")
if support_50_time: print(f"  Support 50% loss at time: {support_50_time:.2f}")
if cortical_50_time: print(f"  Cortical 50% loss at time: {cortical_50_time:.2f}")
if support_10_time: print(f"  Support 90% loss at time: {support_10_time:.2f}")
if cortical_10_time: print(f"  Cortical 90% loss at time: {cortical_10_time:.2f}")

# Check final work per cell values
final_idx = len([x for x in results['cortical_cells'] if x > 0]) - 1
if final_idx > 0:
    print(f"\nFinal work intensity (time {final_idx * 0.01:.2f}):")
    print(f"  Cortical work per cell: {results['work_per_cortical_cell'][final_idx]:.3f}")
    print(f"  Support work per cell: {results['work_per_support_cell'][final_idx]:.3f}")

# Validation
support_fails_first = (support_50_time and cortical_50_time and support_50_time < cortical_50_time)
print(f"\nTIER Predictions:")
print(f"  Support systems fail first: {support_fails_first}")
print(f"  Support works harder per cell: {results['work_per_support_cell'][final_idx] > results['work_per_cortical_cell'][final_idx] if final_idx > 0 else 'Check plot D'}")

print("="*60)