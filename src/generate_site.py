#!/usr/bin/env python3
"""
generate_site.py

Generates a GitHub Pages-ready static site to visualize character relationships,
now using template HTML files for the main index, character pages, group index,
and group pages.

Expects:
- templates/index_template.html
- templates/character_template.html
- templates/groups_index_template.html
- templates/group_template.html

Outputs into docs/:
- docs/global_graph.png
- docs/shepard_diagram.png
- docs/index.html
- docs/characters/<Character>.html
- docs/groups/<GroupName>.html
- docs/images/<subgraph>.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

TEMPLATES_DIR = "templates"
DOCS_DIR = "docs"

# ------------------------------------------------------
# Template Utilities
# ------------------------------------------------------

def load_template(template_name):
    """
    Reads a template file from templates/ and returns it as a string.
    e.g. template_name = 'index_template.html'
    """
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template '{template_name}' not found in {TEMPLATES_DIR}")
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_html(out_path, content):
    """
    Writes the given HTML content to out_path, creating directories if needed.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[HTML] Wrote {out_path}")

# ------------------------------------------------------
# 1. Load data from affinities.txt
# ------------------------------------------------------

def load_affinities(filename="affinities.txt"):
    """Load characters and affinity matrix from a text file with sections:
        # Characters
        Alice
        Bob
        ...
        # Affinities (character1, character2, value)
        Alice,Bob,4
        ...
    Returns (characters, affinity_matrix).
    """
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return [], None
    
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
            elif mode == "affinities":
                parts = line.split(",")
                if len(parts) >= 3:
                    c1, c2, val_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    try:
                        val = float(val_str)
                        affinities_list.append((c1, c2, val))
                    except ValueError:
                        print(f"Could not parse float from '{val_str}'. Skipping line.")
                else:
                    print(f"Malformed affinity line: {line}. Skipping.")
    
    # Build the affinity matrix
    n = len(characters)
    if n < 2:
        print("Not enough characters found.")
        return characters, None
    
    char_to_idx = {c: i for i, c in enumerate(characters)}
    affinity_matrix = np.zeros((n, n))
    
    # Initialize to 0.0, so missing connections remain 0
    for (c1, c2, val) in affinities_list:
        if c1 in char_to_idx and c2 in char_to_idx:
            i, j = char_to_idx[c1], char_to_idx[c2]
            affinity_matrix[i, j] = val
            affinity_matrix[j, i] = val
    
    return characters, affinity_matrix

def affinity_to_distance(affinity_matrix):
    """
    Convert affinity range -5..5 to a distance in [1..6].
    High magnitude => smaller distance => close in MDS
    0 => distance=6 => far in MDS
    """
    return 6 - np.abs(affinity_matrix)

# ------------------------------------------------------
# 2. Plot Global Graph
# ------------------------------------------------------

def plot_global_graph(
    characters,
    affinity_matrix,
    global_img="docs/global_graph.png",
    bg_color="#FFFFFF",      # white background by default
    text_color="#000000",    # black text by default
    font_name="Helvetica"   # can be any installed font
):
    """
    Create a global MDS layout, saving the image to 'global_img'.
    Allows specifying background color, text color, and font.
    """
    from matplotlib import pyplot as plt
    from sklearn.manifold import MDS
    import numpy as np
    import os
    
    # Prepare distance matrix
    dist_mat = affinity_to_distance(affinity_matrix)
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_mat)
    n = len(characters)
    
    # Create dirs if needed
    os.makedirs(os.path.dirname(global_img), exist_ok=True)
    
    # -- Option A: Manually pass facecolor, text color, and set font rcParams
    plt.rcParams["font.family"] = font_name
    
    # We can create the figure with a custom facecolor
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=bg_color)
    
    # If you'd like the axes themselves to share that background:
    ax.set_facecolor(bg_color)
    
    # Plot edges
    for i in range(n):
        for j in range(i+1, n):
            rating = affinity_matrix[i, j]
            if rating == 0:
                continue
            color = '#0c5a6a' if rating > 0 else 'red'
            lw = 0.5 + 4.0 * (abs(rating) / 5.0)
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=color,
                    alpha=1.0,
                    linewidth=lw)
    
    # Plot the points
    ax.scatter(coords[:, 0], coords[:, 1], s=80, c='black')
    
    # Add text labels
    for i, c in enumerate(characters):
        ax.text(
            coords[i, 0], coords[i, 1], c,
            color=text_color,      # override text color
            fontsize=9,
            ha='center', va='bottom',
            bbox=dict(
                facecolor=bg_color,  # If you want the label bubble to match background
                alpha=0.7,
                edgecolor='none'
            )
        )
    
    ax.set_title(
        "",
        color=text_color        # Title color
    )
    ax.axis('off')
    
    # Make sure the figure facecolor is used on save
    plt.savefig(global_img, facecolor=bg_color, dpi=300)
    plt.close()
    print(f"[Global Graph] Saved to {global_img}")


# ------------------------------------------------------
# 3. Template-based Generation
# ------------------------------------------------------

def generate_index_page(characters, global_img="global_graph.png", out_file="docs/index.html"):
    """
    Fills index_template.html with:
      {GLOBAL_GRAPH_IMG} = 'images/global_graph.png' or 'global_graph.png'
      {CHARACTER_LIST}    = <li> items (with optional icons)
    """
    template_str = load_template("index_template.html")
    
    # Build the <li> items
    list_items = []
    for c in characters:
        # optionally use an icon if it exists: images/icons/<lowercase>_icon.png
        icon_path = f"images/icons/{c}_icon.png"
        # If no icon, you might do a fallback. Here we just reference it anyway.
        item_html = f"""
  <li>
    <img src="{icon_path}" alt="{c}" class="char-icon"/>
    <a href="characters/{c}.html">{c}</a>
  </li>
"""
        list_items.append(item_html)
    
    character_list_str = "\n".join(list_items)
    
    print(global_img)
    # Insert placeholders
    filled = template_str.format(
        GLOBAL_GRAPH_IMG=global_img,
        CHARACTER_LIST=character_list_str
    )
    write_html(out_file, filled)

def generate_character_page(characters, affinity_matrix, char_index, epsilon,
                            sub_nodes, sub_img, out_file):
    """
    Fills character_template.html with:
      {CHAR_NAME}, {CONNECTION_COUNT}, {AFFINITY_SUM}, {SUBGRAPH_IMG}, {CONNECTION_TABLE}
    """
    template_str = load_template("character_template.html")
    main_char = characters[char_index]
    
    # Stats: number of connections above ε, sum of all affinity
    row = affinity_matrix[char_index,:]
    n_over_epsilon = sum(
        1 for i, val in enumerate(row) if i!=char_index and abs(val) > epsilon
    )
    total_affinity_sum = sum(
        val for i, val in enumerate(row) if i != char_index
    )
    total_affinity_sum = total_affinity_sum/n_over_epsilon
    
    # Build connection table
    connections = []
    for n_i in sub_nodes:
        if n_i != char_index:
            aff = row[n_i]
            connections.append((characters[n_i], aff))
    connections.sort(key=lambda x: abs(x[1]), reverse=True)
    
    table_rows = []
    for other_char, val in connections:
        val_str = f"{val:+.1f}"
        # link to other_char if needed
        row_html = f"<tr><td><a href='{other_char}.html'>{other_char}</a></td><td>{val_str}</td></tr>"
        table_rows.append(row_html)
    connection_table_html = "\n".join(table_rows)
    
    filled = template_str.format(
        CHAR_NAME=main_char,
        CONNECTION_COUNT=n_over_epsilon,
        AFFINITY_SUM=f"{total_affinity_sum:.2f}",
        SUBGRAPH_IMG=os.path.basename(sub_img),  # e.g. "Alice_graph.png"
        CONNECTION_TABLE=connection_table_html
    )
    write_html(out_file, filled)

# ------------------------------------------------------
# 4. Per-Character MDS & HTML
# ------------------------------------------------------

def plot_character_subgraph(
    characters,
    affinity_matrix,
    char_index,
    epsilon=1.0,
    out_image="docs/images/Unnamed_graph.png",
    bg_color="#FFFFFF",
    text_color="#000000",
    font_name="Helvetica"
):
    """
    Create a subgraph for the main_char with |affinity|>epsilon,
    only edges from main_char to neighbors,
    and let the user specify background, text color, and font.
    """
    from matplotlib import pyplot as plt
    from sklearn.manifold import MDS
    import numpy as np
    import os
    
    main_char = characters[char_index]
    row = affinity_matrix[char_index, :]
    
    neighbors_idx = [i for i, val in enumerate(row) if i != char_index and abs(val) > epsilon]
    if not neighbors_idx:
        sub_nodes = [char_index]
    else:
        sub_nodes = [char_index] + neighbors_idx
    
    sub_aff = affinity_matrix[sub_nodes, :][:, sub_nodes]
    sub_dist = 6 - np.abs(sub_aff)
    
    if len(sub_nodes) == 1:
        coords = np.array([[0, 0]])
    else:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(sub_dist)
        coords = coords - coords[0]  # center the main char at (0,0)
    
    os.makedirs(os.path.dirname(out_image), exist_ok=True)
    
    # Set style
    plt.rcParams["font.family"] = font_name
    
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    n_sub = len(sub_nodes)
    for i in range(1, n_sub):
        rating = sub_aff[0, i]
        if rating == 0:
            continue
        color = '#0c5a6a' if rating > 0 else 'red'
        lw = 0.5 + 4.0 * (abs(rating) / 5.0)
        ax.plot([coords[0, 0], coords[i, 0]],
                [coords[0, 1], coords[i, 1]],
                color=color,
                linewidth=lw,
                alpha=1.0)
    
    ax.scatter(coords[:, 0], coords[:, 1], s=80, c='black')
    for idx, node_idx in enumerate(sub_nodes):
        c_name = characters[node_idx]
        if node_idx == char_index:
            facec = '#0c5a6a'
        else:
            facec = bg_color
        ax.text(
            coords[idx, 0], coords[idx, 1], c_name,
            color=text_color,
            fontsize=9,
            ha='center', va='bottom',
            bbox=dict(facecolor=facec, alpha=0.7, edgecolor='none')
        )
    
    ax.set_title(f"{main_char}'s Connections", color=text_color)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_image, facecolor=bg_color, dpi=300)
    plt.close()
    
    return sub_nodes

# ------------------------------------------------------
# 5. Groups
# ------------------------------------------------------

def parse_groups(filename="groups.txt"):
    """
    Each line:
      Group Name, Member1, Member2, ...
    Returns list of (group_name, [member1, member2, ...]).
    """
    groups = []
    if not os.path.exists(filename):
        print(f"No groups file '{filename}'. Skipping group pages.")
        return groups
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts)<2:
                print(f"Bad group line: {line}")
                continue
            group_name = parts[0]
            members = parts[1:]
            groups.append((group_name, members))
    return groups

def plot_group_subgraph(
    group_name,
    group_members,
    characters,
    affinity_matrix,
    epsilon=1.0,
    out_image="docs/images/UnnamedGroup_graph.png",
    bg_color="#12111c",
    text_color="#a7eaee",
    font_name="Helvetica"
):
    """
    1) Identify neighbors that have |affinity|>ε with *all* group_members.
    2) Combine group_members + those neighbors.
    3) Run MDS on that subset.
    4) Plot only edges among group members or group->neighbor edges.
    5) Use bg_color, text_color, and font_name for styling.
    """
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    from sklearn.manifold import MDS

    char_to_idx = {c: i for i, c in enumerate(characters)}
    group_indices = []
    for gm in group_members:
        if gm in char_to_idx:
            group_indices.append(char_to_idx[gm])
        else:
            print(f"Warning: '{gm}' not found in characters. Skipping.")
    if not group_indices:
        return []

    # Identify neighbors
    n_chars = len(characters)
    neighbors = []
    for c_idx in range(n_chars):
        if c_idx in group_indices:
            continue
        # Must be strongly connected to *all* in group
        is_strong = all(abs(affinity_matrix[g, c_idx]) > epsilon for g in group_indices)
        if is_strong:
            neighbors.append(c_idx)

    sub_nodes = group_indices + neighbors
    if not sub_nodes:
        return []

    # Build sub-affinity / sub-dist
    sub_aff = affinity_matrix[sub_nodes, :][:, sub_nodes]
    sub_dist = 6 - np.abs(sub_aff)

    # MDS unless only 1 node
    if len(sub_nodes) == 1:
        coords = np.array([[0, 0]])
    else:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(sub_dist)

    # Center the group by shifting centroid of group_indices
    gsize = len(group_indices)
    group_coords = coords[:gsize, :]
    centroid = group_coords.mean(axis=0)
    coords = coords - centroid

    # Create output dir
    os.makedirs(os.path.dirname(out_image), exist_ok=True)

    # Set font via rcParams
    plt.rcParams["font.family"] = font_name

    # Create figure with background color
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Distinguish group vs neighbors
    n_sub = len(sub_nodes)
    for i in range(n_sub):
        for j in range(i+1, n_sub):
            both_group = (i < gsize and j < gsize)
            group_neighbor = (i < gsize <= j) or (j < gsize <= i)
            # Only draw edges among group members or from group->neighbor
            if not (both_group or group_neighbor):
                continue
            rating = sub_aff[i, j]
            if abs(rating) <= epsilon:
                continue
            color = '#0c5a6a' if rating > 0 else 'red'
            lw = 0.5 + 4.0 * (abs(rating) / 5.0)
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=color, linewidth=lw, alpha=1.0)

    # Scatter
    ax.scatter(coords[:, 0], coords[:, 1], s=80, c='black')

    # Labels
    for idx, node_idx in enumerate(sub_nodes):
        c_name = characters[node_idx]
        if idx < gsize:
            facec = '#0c5a6a'  # highlight group members
        else:
            facec = bg_color
        ax.text(
            coords[idx, 0],
            coords[idx, 1],
            c_name,
            color=text_color,
            fontsize=9,
            ha='center',
            va='bottom',
            bbox=dict(facecolor=facec, alpha=0.7, edgecolor='none')
        )

    ax.set_title(f"Group: {group_name}", color=text_color)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_image, facecolor=bg_color, dpi=300)
    plt.close()
    return sub_nodes


def generate_group_page(group_name, group_members, sub_nodes,
                        characters, affinity_matrix, epsilon=1.0,
                        sub_img="docs/images/UnnamedGroup_graph.png",
                        out_file="docs/groups/UnnamedGroup.html"):
    """
    Fills group_template.html:
      {GROUP_NAME}, {GROUP_MEMBERS}, {GROUP_IMG}, {NEIGHBOR_TABLE}
    """
    template_str = load_template("group_template.html")
    safe_name = group_name.replace(" ","_")
    
    # Build neighbor table (neighbors = sub_nodes minus group_indices)
    char_to_idx = {c:i for i,c in enumerate(characters)}
    group_indices = [char_to_idx[m] for m in group_members if m in char_to_idx]
    neighbor_indices = [n for n in sub_nodes if n not in group_indices]
    
    rows = []
    for ni in neighbor_indices:
        name = characters[ni]
        affs = [affinity_matrix[gi, ni] for gi in group_indices]
        if affs:
            avg_aff = np.mean(affs)
            avg_mag = np.mean([abs(a) for a in affs])
        else:
            avg_aff = 0
            avg_mag = 0
        row_html = f"<tr><td>{name}</td><td>{avg_aff:+.2f}</td><td>{avg_mag:.2f}</td></tr>"
        rows.append(row_html)
    neighbor_table = "\n".join(rows)
    
    group_members_str = ", ".join(group_members)
    
    filled = template_str.format(
        GROUP_NAME=group_name,
        GROUP_MEMBERS=group_members_str,
        GROUP_IMG=os.path.basename(sub_img),
        NEIGHBOR_TABLE=neighbor_table
    )
    write_html(out_file, filled)

def generate_groups_index(groups, out_file="docs/groups/index.html"):
    """
    Fills groups_index_template.html with {GROUP_LINKS}.
    """
    template_str = load_template("groups_index_template.html")
    links_html = []
    for g_name, _ in groups:
        safe_name = g_name.replace(" ","_")
        links_html.append(f'<li><a href="{safe_name}.html">{g_name}</a></li>')
    group_links = "\n".join(links_html)
    
    filled = template_str.format(GROUP_LINKS=group_links)
    write_html(out_file, filled)

def generate_all_groups(characters, affinity_matrix, groups_txt="groups.txt",
                        epsilon=1.0, out_dir="docs/groups", bg_color="#FFFFFF",
                        text_color="#000000", font_name="sans-serif"):
    groups = parse_groups(groups_txt)
    if not groups:
        return
    
    for (g_name, members) in groups:
        safe_name = g_name.replace(" ","_")
        sub_img = os.path.join("docs","images",f"{safe_name}_graph.png")
        sub_nodes = plot_group_subgraph(g_name, members,
                                        characters, affinity_matrix,
                                        epsilon=epsilon,
                                        out_image=sub_img)
        out_file = os.path.join(out_dir, f"{safe_name}.html")
        generate_group_page(g_name, members, sub_nodes, characters, affinity_matrix,
                            epsilon=epsilon, sub_img=sub_img, out_file=out_file)
    
    # index
    idx_file = os.path.join(out_dir, "index.html")
    generate_groups_index(groups, out_file=idx_file)

# ------------------------------------------------------
# 6. Main Orchestration
# ------------------------------------------------------

def main():
    # 1) Load the data
    characters, affinity_matrix = load_affinities("affinities.txt")
    if not characters or affinity_matrix is None:
        print("Failed to load data from affinities.txt. Exiting.")
        return
    if len(characters) < 2:
        print("Not enough characters. Exiting.")
        return

    # ------------------------------------------------------------------
    # NEW: Define your desired style parameters for the plots
    # ------------------------------------------------------------------
    bg_color   = "#12111c"   # dark background
    text_color = "#a7eaee"   # light text
    font_name = "Helvetica"     # choose any installed font

    # 2) Plot global graph
    plot_global_graph(
        characters,
        affinity_matrix,
        global_img="docs/global_graph.png",
        bg_color=bg_color,
        text_color=text_color,
        font_name=font_name
    )

    # 3) Generate the main index page
    generate_index_page(
        characters,
        global_img="global_graph.png",
        out_file="docs/index.html"
    )

    # 4) Generate each character's subgraph & page
    epsilon = 0.0
    for i, c in enumerate(characters):
        sub_img = f"docs/images/{c}_graph.png"

        sub_nodes = plot_character_subgraph(
            characters,
            affinity_matrix,
            i,
            epsilon,
            out_image=sub_img,
            bg_color=bg_color,
            text_color=text_color,
            font_name=font_name
        )

        out_file = f"docs/characters/{c}.html"
        generate_character_page(
            characters,
            affinity_matrix,
            i,
            epsilon,
            sub_nodes,
            sub_img,
            out_file
        )

    # 5) Generate group pages/index
    generate_all_groups(
        characters,
        affinity_matrix,
        groups_txt="groups.txt",
        epsilon=epsilon,
        out_dir="docs/groups",
        bg_color=bg_color,
        text_color=text_color,
        font_name=font_name
    )

    print("[DONE] Site generation complete! Check 'docs/' folder.")


if __name__ == "__main__":
    main()