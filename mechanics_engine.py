import numpy as np
from scipy.optimize import minimize, fsolve
 
# === FOUR-BAR LINKAGE KINEMATICS ===
def fourbar_position(theta2, L1, L2, L3, L4):
    """Solve four-bar position analysis using Freudenstein equations."""
    # Coupler point at (xC, yC), output crank at theta4
    # Newton-Raphson on closure equations
    
    A = 2*L1*L4 - 2*L2*L4*np.cos(theta2)
    B = -2*L2*L4*np.sin(theta2)
    C = L1**2 + L2**2 + L4**2 - L3**2 - 2*L1*L2*np.cos(theta2)
    
    discriminant = A**2 + B**2 - C**2
    if discriminant < 0:
        return None  # not assemblable
    
    theta4 = 2*np.arctan2(B - np.sqrt(discriminant), A + C)
    
    # Coupler point (midpoint of L3)
    xB = L2*np.cos(theta2)
    yB = L2*np.sin(theta2)
    xD = L1 + L4*np.cos(theta4)
    yD = L4*np.sin(theta4)
    xC = (xB + xD) / 2
    yC = (yB + yD) / 2
    
    return {"theta4": theta4, "xB": xB, "yB": yB, "xD": xD, "yD": yD, "xC": xC, "yC": yC}
 
def simulate_fourbar(L1, L2, L3, L4, n_steps=120):
    """Simulate full revolution of crank L2."""
    angles = np.linspace(0, 2*np.pi, n_steps)
    coupler_path = []
    
    for theta in angles:
        result = fourbar_position(theta, L1, L2, L3, L4)
        if result is not None:
            coupler_path.append([result["xC"], result["yC"]])
    
    return np.array(coupler_path), angles
 
def grashof_check(L1, L2, L3, L4):
    """Check Grashof's condition for full rotation."""
    lengths = sorted([L1, L2, L3, L4])
    s = lengths[0]
    l = lengths[3]
    p = lengths[1]
    q = lengths[2]
    return s + l <= p + q  # Grashof condition
 
# === PARAMETER IDENTIFICATION ===
def spring_mass_damper(t, params, F0=1.0, omega=2*np.pi):
    """Forced spring-mass-damper response."""
    m, c, k = params
    # Frequency response amplitude
    omega_n = np.sqrt(k/m)
    zeta = c / (2*np.sqrt(k*m))
    r = omega/omega_n
    H = 1 / np.sqrt((1-r**2)**2 + (2*zeta*r)**2)
    phase = np.arctan2(-2*zeta*r, 1-r**2)
    return (F0/k) * H * np.cos(omega*t + phase)
 
def identify_parameters(t_data, x_data, F0=1.0, omega=2*np.pi):
    """Least-squares identification of m, c, k from noisy data."""
    def residual(params):
        x_pred = spring_mass_damper(t_data, params, F0, omega)
        return np.sum((x_pred - x_data)**2)
    
    # Initial guess
    x0 = [1.0, 0.5, 50.0]
    bounds = [(0.1, 10), (0.01, 5), (1, 500)]
    result = minimize(residual, x0, method='L-BFGS-B', bounds=bounds)
    return result.x  # [m, c, k]
 
# === TOPOLOGY OPTIMIZATION (Simplified SIMP) ===
def topology_optimize_2d(nx=20, ny=10, volfrac=0.5, n_iter=30):
    """Simplified topology optimization for 2D cantilever beam."""
    # Density field initialization
    x = np.ones((ny, nx)) * volfrac
    
    # Simple gradient-descent inspired SIMP
    # In a real SIMP this would use FEM compliance + sensitivity analysis
    # Here we approximate: keep density highest near load and supports
    
    history = []
    for it in range(n_iter):
        # Sensitivity (simplified): higher density toward load (right side, mid-height)
        load_x, load_y = nx-1, ny//2
        support_x = 0
        
        sensitivity = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                # Distance from load + support beam path
                d_load = np.sqrt((j-load_x)**2 + (i-load_y)**2)
                d_support = j  # distance from x=0
                
                # Sensitivity: higher near direct load path
                sensitivity[i,j] = 1.0 / (1 + 0.1*d_load + 0.05*d_support) + np.random.uniform(0, 0.05)
        
        # Update density (gradient descent with volume constraint)
        x_new = x + 0.05 * (sensitivity - sensitivity.mean())
        x_new = np.clip(x_new, 0.001, 1.0)
        
        # Volume preservation: rescale
        current_vol = x_new.sum()
        target_vol = volfrac * nx * ny
        x_new *= target_vol / current_vol
        x_new = np.clip(x_new, 0.001, 1.0)
        
        x = x_new
        history.append(x.copy())
    
    return x, history
 
if __name__ == "__main__":
    # Test 4-bar linkage
    L1, L2, L3, L4 = 100, 30, 80, 60
    grashof = grashof_check(L1, L2, L3, L4)
    print(f"Grashof condition: {'PASS - full rotation possible' if grashof else 'FAIL'}")
    
    path, angles = simulate_fourbar(L1, L2, L3, L4)
    print(f"Coupler path points: {len(path)}")
    print(f"Path range x: [{path[:,0].min():.2f}, {path[:,0].max():.2f}]")
    print(f"Path range y: [{path[:,1].min():.2f}, {path[:,1].max():.2f}]")
    
    # Test parameter ID
    np.random.seed(42)
    t = np.linspace(0, 2, 100)
    true_params = [2.0, 0.5, 100.0]  # m, c, k
    x_clean = spring_mass_damper(t, true_params)
    x_noisy = x_clean + np.random.normal(0, 0.005, len(t))
    identified = identify_parameters(t, x_noisy)
    print(f"\nTrue params: m={true_params[0]}, c={true_params[1]}, k={true_params[2]}")
    print(f"Identified:  m={identified[0]:.3f}, c={identified[1]:.3f}, k={identified[2]:.3f}")
    
    # Test topology
    x_final, _ = topology_optimize_2d(nx=20, ny=10, volfrac=0.4, n_iter=20)
    print(f"\nTopology opt: final volume fraction = {x_final.mean():.3f}")