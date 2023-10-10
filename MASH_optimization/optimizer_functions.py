import sympy as sp

k12,k13,k23 = sp.symbols("k12 k13 k23")
p1,p2,p3 = sp.symbols("p1 p2 p3") 
matrix = sp.Matrix([[-p3*k13-p2*k12, p1*k12, p1*k13],[p2*k12, -p1*k12-p3*k23, p2*k23], [p3*k13, p3*k23, -p1*k13-p2*k23]])

eigenvalues = matrix.eigenvals()

solution_found = False
for k_value in sp.solvers.solve(eigenvalues.values(), [k12, k13, k23]):
    if all(k >= 0 for k in k_value.values()):
        solution_found = True
        print("Non-negative values of k that make eigenvalues positive:")
        print(k_value)

if not solution_found:
    print("No non-negative values of k found that make any eigenvalue positive.")