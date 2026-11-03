from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from numpy import cross, dot, arccos
from numpy.linalg import norm
import numpy as np

# Load the structure from the POSCAR file
structure = Structure.from_file("POSCAR")

# Initialize the SpacegroupAnalyzer with desired symmetry precision
symmetry_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

# Get the space group symbol and number
space_group_symbol = symmetry_analyzer.get_space_group_symbol()
space_group_number = symmetry_analyzer.get_space_group_number()

# Get the symmetry and Wyckoff positions
symmetry_data = symmetry_analyzer.get_symmetry_dataset()
wyckoff_positions = symmetry_data["wyckoffs"]

# Display results
print("Space Group Symbol:", space_group_symbol)
print("Space Group Number:", space_group_number)
print("Wyckoff Positions:", wyckoff_positions)
#quit()
# Use the SpacegroupAnalyzer if needed
spa = SpacegroupAnalyzer(structure, symprec=0.1)

# Get the primitive standard structure (if required for your purpose)
ordered_structure = spa.get_primitive_standard_structure()
#ordered_structure = spa.get_conventional_standard_structure()
analyzer = SpacegroupAnalyzer(ordered_structure)
point_group = analyzer.get_point_group_symbol()

# Access the lattice parameters a, b, c
a, b, c = ordered_structure.lattice.a, ordered_structure.lattice.b, ordered_structure.lattice.c

# Access the lattice vectors
a_vec = ordered_structure.lattice.matrix[0]  # Lattice vector 'a'
b_vec = ordered_structure.lattice.matrix[1]  # Lattice vector 'b'
c_vec = ordered_structure.lattice.matrix[2]  # Lattice vector 'c'

# Compute the normal to the 'ab' plane
normal_ab_plane = cross(a_vec, b_vec)

# Calculate the angle between the normal_ab_plane and c_vec
cos_theta_ab = dot(a_vec, b_vec) / (norm(a_vec) * norm(b_vec))
angle_ab = np.degrees(arccos(cos_theta_ab))
cos_theta = dot(normal_ab_plane, c_vec) / (norm(normal_ab_plane) * norm(c_vec))
angle = np.degrees(arccos(cos_theta))

# Access the angles alpha, beta, gamma
alpha = ordered_structure.lattice.alpha
beta = ordered_structure.lattice.beta
gamma = ordered_structure.lattice.gamma

print("Lattice Angles - alpha:", alpha, "beta:", beta, "gamma:", gamma)
print("Angle between 'ab' plane normal and c-axis:", angle)

# Print each atom index, element, and Wyckoff position
print("\nAtomic Sites and Wyckoff Positions:")
for i, site in enumerate(ordered_structure.sites):
    element = site.species_string
    wyckoff = wyckoff_positions[i]
    print(f"Index: {i}, Element: {element}, Wyckoff Position: {wyckoff}")

