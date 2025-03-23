import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def get_input_method():
    """Ask the user to choose between manual input or loading from a file."""
    while True:
        print("How would you like to input the list of characters?")
        print("1. Manual input")
        print("2. Load from file")
        print("3. Use existing affinities from affinities.txt")
        choice = input("Enter your choice (1, 2, or 3): ")
        
        if choice in ["1", "2", "3"]:
            return choice
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def get_characters_from_input():
    """Get characters from manual user input."""
    print("Enter your list of characters (one per line). Type 'done' when finished:")
    characters = []
    while True:
        character = input().strip()
        if character.lower() == 'done':
            break
        if character:
            characters.append(character)
    return characters

def get_characters_from_file():
    """Get characters from a file specified by the user."""
    while True:
        filename = input("Enter the filename containing your characters (one per line): ")
        try:
            with open(filename, 'r') as file:
                characters = [line.strip() for line in file if line.strip()]
                
            if not characters:
                print("The file is empty. Please specify a different file.")
                continue
                
            print(f"Successfully loaded {len(characters)} characters from '{filename}':")
            for character in characters:
                print(f"- {character}")
            
            return characters
        except FileNotFoundError:
            print(f"File '{filename}' not found. Please check the filename and try again.")
        except Exception as e:
            print(f"Error reading file: {e}")

def get_characters():
    """Get characters from user's preferred input method."""
    choice = get_input_method()
    if choice == "1":
        return get_characters_from_input(), choice
    elif choice == "2":
        return get_characters_from_file(), choice
    else:  # choice == "3"
        return None, choice

def create_affinity_matrix(characters):
    """
    Create an affinity matrix based on user input, 
    range: -5 (intense hatred) to +5 (insane love), 0 = no relationship.
    """
    n = len(characters)
    affinity_matrix = np.zeros((n, n))
    
    print("\nRate the relationship between each pair of characters (-5 to 5):")
    print("-5 = intense hatred, +5 = insane love, 0 = no relationship")
    
    for i in range(n):
        # Diagonal = 0, i.e. self-relationship is zero
        affinity_matrix[i, i] = 0
        for j in range(i+1, n):
            while True:
                try:
                    prompt = f"Relationship between '{characters[i]}' and '{characters[j]}' (-5 to 5): "
                    rating = float(input(prompt))
                    if -5 <= rating <= 5:
                        break
                    else:
                        print("Please enter a number between -5 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
            
            affinity_matrix[i, j] = rating
            affinity_matrix[j, i] = rating  # Symmetric
    
    # Save affinities to file
    save_affinities_to_file(characters, affinity_matrix)
    
    return affinity_matrix

def save_affinities_to_file(characters, affinity_matrix, filename="affinities.txt"):
    """Save character affinities to a file."""
    with open(filename, 'w') as file:
        # Write characters section
        file.write("# Characters\n")
        for character in characters:
            file.write(f"{character}\n")
        
        # Write affinities section
        file.write("\n# Affinities (character1, character2, affinity_value)\n")
        n = len(characters)
        for i in range(n):
            for j in range(i+1, n):
                file.write(f"{characters[i]},{characters[j]},{affinity_matrix[i, j]}\n")
    
    print(f"Affinities saved to {filename}")

def load_affinities_from_file(filename="affinities.txt"):
    """Load character affinities from a file."""
    try:
        characters = []
        affinities_list = []
        mode = None
        
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("#"):
                    if "Characters" in line:
                        mode = "characters"
                    elif "Affinities" in line:
                        mode = "affinities"
                    continue
                
                if mode == "characters":
                    characters.append(line)
                elif mode == "affinities" and "," in line:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        char1, char2, value = parts[0], parts[1], float(parts[2])
                        affinities_list.append((char1, char2, value))
        
        if not characters or len(characters) < 2:
            print("Not enough characters found in the file.")
            return None, None
            
        # Create character to index mapping
        char_to_idx = {character: i for i, character in enumerate(characters)}
        
        # Initialize affinity matrix
        n = len(characters)
        affinity_matrix = np.zeros((n, n))
        
        # Fill affinity matrix
        for char1, char2, value in affinities_list:
            if char1 in char_to_idx and char2 in char_to_idx:
                i, j = char_to_idx[char1], char_to_idx[char2]
                affinity_matrix[i, j] = value
                affinity_matrix[j, i] = value  # Symmetric
        
        print(f"Loaded affinities for {n} characters from {filename}")
        return characters, affinity_matrix
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading affinities: {e}")
        return None, None

def affinity_to_distance(affinity_matrix):
    """
    Convert affinity (range -5..5) to a distance in [1..6].
    - Larger magnitude => smaller distance (±5 => distance=1).
    - Near zero => bigger distance (0 => distance=6).
    """
    distance_matrix = 6 - np.abs(affinity_matrix)
    return distance_matrix

def plot_mds(characters, distance_matrix, affinity_matrix):
    """Create and save an MDS plot and visually distinguish positive vs negative edges,
       with edge thickness corresponding to magnitude."""
    from sklearn.manifold import MDS
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    
    stress = mds.stress_
    print(f"MDS Stress value: {stress:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=50)
    
    # Label each point with the character name
    for i, character in enumerate(characters):
        ax.annotate(character, (coordinates[i, 0], coordinates[i, 1]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Draw edges with different colors for + vs. - and different thickness for magnitude
    n = len(characters)
    for i in range(n):
        for j in range(i+1, n):
            rating = affinity_matrix[i, j]
            if rating == 0:
                # Skip zero relationships or optionally draw them differently
                continue
            edge_color = 'green' if rating > 0 else 'red'
            
            # Map relationship magnitude (abs(rating)) to line width
            # e.g., abs(rating)=5 => lw ~ 4.5, abs(rating)=1 => lw ~ 1.3, etc.
            lw = 0.5 + 4.0 * (abs(rating) / 5.0)
            
            ax.plot(
                [coordinates[i, 0], coordinates[j, 0]],
                [coordinates[i, 1], coordinates[j, 1]],
                color=edge_color,
                alpha=1.0,
                linewidth=lw
            )
    
    ax.set_title('Character Relationship Map (+/- shown by edge color, thickness = magnitude)')
    plt.savefig('mds_sign_edges_thickness.png')
    plt.show()
    
    print("Character map with sign-colored, magnitude-thickness edges saved as 'mds_sign_edges_thickness.png'")
    
    # Shepard diagram (optional)...
    fig, ax2 = plt.subplots(figsize=(6,6))
    mds_distances = squareform(pdist(coordinates))
    orig_dist_flat, mds_dist_flat = [], []
    for i in range(n):
        for j in range(i+1, n):
            orig_dist_flat.append(distance_matrix[i, j])
            mds_dist_flat.append(mds_distances[i, j])
    correlation, _ = pearsonr(orig_dist_flat, mds_dist_flat)
    
    ax2.scatter(orig_dist_flat, mds_dist_flat, alpha=0.6)
    ax2.set_xlabel('Original Distances')
    ax2.set_ylabel('MDS Distances')
    ax2.set_title(f'Shepard Diagram\nCorrelation: {correlation:.4f}')
    
    min_val = min(min(orig_dist_flat), min(mds_dist_flat))
    max_val = max(max(orig_dist_flat), max(mds_dist_flat))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('shepard_diagram.png')
    plt.show()
    
    print("Shepard diagram saved as 'shepard_diagram.png'")

def main():
    # Step 1: Get characters and affinities
    characters, choice = get_characters()
    
    if choice == "3":
        # Load characters and affinities from file
        characters, affinity_matrix = load_affinities_from_file()
        if characters is None or len(characters) < 2:
            print("Failed to load valid affinities. Exiting.")
            return
    else:
        if characters is None or len(characters) < 2:
            print("Need at least 2 characters to create a relationship map.")
            return
        
        # Create affinity matrix
        affinity_matrix = create_affinity_matrix(characters)
    
    # Step 2: Convert to distance matrix
    distance_matrix = affinity_to_distance(affinity_matrix)
    
    # Step 3: Apply MDS and visualize
    plot_mds(characters, distance_matrix, affinity_matrix)

if __name__ == "__main__":
    main()