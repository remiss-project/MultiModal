import networkx as nx
import matplotlib.pyplot as plt
import random
import pdb
def color_mapped_nodes(graph1, graph2, mapping2,graph3,mapping3):
    # Create new graphs initialized with nodes from the original graphs
    new_graph1 = graph1.copy()
    new_graph2 = graph2.copy()
    new_graph3 = graph3.copy()
    
    # Assign colors to nodes and edges in new graphs
    for node in new_graph1.nodes:
        new_graph1.nodes[node]['color'] = 'white'  # Assign a default color
    for node in new_graph2.nodes:
        new_graph2.nodes[node]['color'] = 'white'  # Assign a default color
    for node in new_graph3.nodes:
        new_graph3.nodes[node]['color'] = 'white'  # Assign a default color        
    for edge in new_graph1.edges:
        new_graph1.edges[edge]['color'] = 'black'  # Assign a default color
    for edge in new_graph2.edges:
        new_graph2.edges[edge]['color'] = 'black'  # Assign a default color
    for edge in new_graph3.edges:
        new_graph3.edges[edge]['color'] = 'black'  # Assign a default color        
        
    
    # Generate a list of unique colors
    distinct_colors = [(random.random(), random.random(), random.random()) for _ in range(100)]  # Choose any desired number of colors
    
    # Assign distinct colors to mapped nodes and their corresponding counterparts
    for idx, edge_mapping in mapping2.items():
        src1 = edge_mapping['src1']
        src2 = edge_mapping['src2']
        tar1 = edge_mapping['tar1']
        tar2 = edge_mapping['tar2']
        
        # Use a distinct color for this mapping
        color = distinct_colors.pop()  # Remove a color from the list
        
        # Assign the color to the corresponding nodes in both graphs
        new_graph1.nodes[src1]['color'] = color
        new_graph2.nodes[tar1]['color'] = color
        
        # Find a different color for the other end of the edge
        colorb = distinct_colors.pop()  # Remove a color from the list
        
        # Assign the color to the corresponding nodes in both graphs
        new_graph1.nodes[src2]['color'] = colorb
        new_graph2.nodes[tar2]['color'] = colorb
        
        colorc = distinct_colors.pop()
        
        # Color the corresponding edges in both graphs
        new_graph1.edges[src1, src2]['color'] = colorc
        new_graph2.edges[tar1, tar2]['color'] = colorc

    

    # Assign distinct colors to mapped nodes and their corresponding counterparts
    for idx, edge_mapping in mapping3.items():
        src1 = edge_mapping['src1']
        src2 = edge_mapping['src2']
        tar1 = edge_mapping['tar1']
        tar2 = edge_mapping['tar2']
        
        # Use a distinct color for this mapping
        
        if new_graph1.nodes[src1]['color'] is not None: ##NODE HAS BEEN COLORED COPY
           new_graph3.nodes[tar1]['color']  = new_graph1.nodes[src1]['color']
        else:
            color = distinct_colors.pop()  # Remove a color from the list
            new_graph1.nodes[src1]['color'] = color
            new_graph3.nodes[tar1]['color'] = color
        
        if new_graph1.nodes[src2]['color'] is not None:
            new_graph3.nodes[tar2]['color']  =new_graph1.nodes[src2]['color']
        else:     
            colorb = distinct_colors.pop()  # Remove a color from the list
            # Assign the color to the corresponding nodes in both graphs
            new_graph1.nodes[src2]['color'] = colorb
            new_graph3.nodes[tar2]['color'] = colorb
        
        # Color the corresponding edges in both graphs
        if new_graph1.edges[src1, src2]['color']  is not None:
           new_graph3.edges[tar1, tar2]['color'] = new_graph1.edges[src1, src2]['color']
        else:
           colorc = distinct_colors.pop()    
           new_graph1.edges[src1, src2]['color'] = colorc
           new_graph3.edges[tar1, tar2]['color'] = colorc
     
    
    
    
    return new_graph1, new_graph2,new_graph3

# Example usage
# Assuming CDG, VDG, TDG are the original graphs and vmaps is the mapping dictionary

# Create original graphs (Replace CDG, VDG, TDG with your actual graphs)
CDG = nx.Graph()
CDG.add_nodes_from(['Photographs', 'Thursday', 'Air Force Intelligence', 'Headquarters', 'Aleppo'])
CDG.add_edges_from([('Photographs', 'Thursday'), ('Photographs', 'Air Force Intelligence'), 
                    ('Air Force Intelligence', 'Headquarters'), ('Air Force Intelligence', 'Aleppo')])

VDG = nx.Graph()
VDG.add_nodes_from(['Photographs', 'Thursday', 'Air Force Intelligence', 'Aleppo'])
VDG.add_edges_from([('Photographs', 'Thursday'), ('Photographs', 'Air Force Intelligence'), 
                    ('Air Force Intelligence', 'Aleppo')])

TDG = nx.Graph()
TDG.add_nodes_from(['Photographs', 'Thursday', 'Air Force Intelligence', 'Headquarters', 'Aleppo'])
TDG.add_edges_from([('Photographs', 'Thursday'), ('Photographs', 'Air Force Intelligence'), 
                    ('Air Force Intelligence', 'Headquarters'), ('Air Force Intelligence', 'Aleppo')])


# Mapping dictionary
vmaps = {
    0: {'src1': 'Photographs', 'src2': 'Thursday', 'tar1': 'Photographs', 'tar2': 'Thursday'},
    1: {'src1': 'Photographs', 'src2': 'Air Force Intelligence', 'tar1': 'Photographs', 'tar2': 'Air Force Intelligence'},
    2: {'src1': 'Air Force Intelligence', 'src2': 'Aleppo', 'tar1': 'Air Force Intelligence', 'tar2': 'Aleppo'}
}
tmaps={ 0: {'src1': 'Photographs', 'src2': 'Thursday', 'tar1': 'Photographs', 'tar2': 'Thursday'}, 1: {'src1': 'Photographs', 'src2': 'Air Force Intelligence', 'tar1': 'Photographs', 'tar2': 'Air Force Intelligence'}, 2: {'src1': 'Air Force Intelligence', 'src2': 'Headquarters', 'tar1': 'Air Force Intelligence', 'tar2': 'Headquarters'}, 3: {'src1': 'Air Force Intelligence', 'src2': 'Aleppo', 'tar1': 'Air Force Intelligence', 'tar2': 'Aleppo'}}
# Color the mapped nodes, their corresponding counterparts, and edges
new_CDG, new_VDG ,new_TDG= color_mapped_nodes(CDG, VDG, vmaps,TDG,tmaps)

# Save the graphs as images
plt.figure(figsize=(12, 5))
nx.draw(new_CDG, with_labels=True, node_color=[new_CDG.nodes[n]['color'] for n in new_CDG.nodes()], 
        edge_color=[new_CDG.edges[e]['color'] for e in new_CDG.edges()], cmap=plt.cm.tab10)
plt.savefig('new_CDG.png')

plt.figure(figsize=(12, 5))
nx.draw(new_VDG, with_labels=True, node_color=[new_VDG.nodes[n]['color'] for n in new_VDG.nodes()], 
        edge_color=[new_VDG.edges[e]['color'] for e in new_VDG.edges()], cmap=plt.cm.tab10)
plt.savefig('new_VDG.png')


plt.figure(figsize=(12, 5))
nx.draw(new_TDG, with_labels=True, node_color=[new_TDG.nodes[n]['color'] for n in new_TDG.nodes()], 
        edge_color=[new_TDG.edges[e]['color'] for e in new_TDG.edges()], cmap=plt.cm.tab10)
plt.savefig('new_TDG.png')



