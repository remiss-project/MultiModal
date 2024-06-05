import torch
import networkx as nx
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity  as pw_cosine_similarity
from scipy.optimize import linear_sum_assignment
import os
from transformers import BertTokenizer, BertModel
import pdb
import matplotlib.pyplot as plt
from utils.gkb import search_gkb,search_gkb_topk
import numpy as np
import io
import glob
import ast
import shutil
import json
import string
import copy
import re
import random
from PIL import Image, ImageDraw
from utils.config import dictfile,clean_dictfile,DATA_DIR,cprompt,res_folder,dataset_name,gkb_thres,llm,txt_sim_thres
if llm=='chatgpt':
    from utils.chtgpt import get_response_ch_api_wrap
from  utils.graph_rules_all import graph_format_ex, graph_format_template,graph_rules_combined,graph_examples,create_graph_from_output,er_to_graph_code

def create_empty_image(width, height, background_color=(255, 255, 255)):
    # Create a new image with the specified width, height, and background color
    image = Image.new("RGB", (width, height), background_color)
    return image

print(dataset_name)
if dataset_name!='isolated':
   a=np.load(dictfile,allow_pickle=True)
   chatgpt_res_dict=a[a.files[0]]
   chatgpt_res_dict=chatgpt_res_dict.flat[0]

   a=np.load(clean_dictfile,allow_pickle=True)
   clean_chatgpt_res_dict=a[a.files[0]]
   clean_chatgpt_res_dict=clean_chatgpt_res_dict.flat[0]


empty_dg       = nx.Graph()
empty_dg_img   = create_empty_image(200,200)    
empty_er_graph = {
    'nodes': [('unk1', {'ent_type': 'unk', 'data': 'unk'}),
              ('unk2', {'ent_type': 'unk', 'data': 'unk'})
             ],
    'edges': [('unk1', 'unk2', {'action': 'unk'})
             ]
                }   
                
                

if dataset_name=='remiss':
   tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')#('bert-base-uncased')
   bertmodel = BertModel.from_pretrained('bert-base-multilingual-uncased')#('bert-base-uncased')
else:
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#('bert-base-uncased')
   bertmodel = BertModel.from_pretrained('bert-base-uncased')#('bert-base-uncased')


#----------------------------------------------------------------------------------------------------------#

def color_entire_walk(graph, node1, node2, color):
    try:
        # Find the shortest path (walk) between node1 and node2
        shortest_path = nx.shortest_path(graph, source=node1, target=node2)

        # Color all edges along the shortest path
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            graph.edges[u, v]['color'] = color
    except nx.NetworkXNoPath:
        # If no path exists between node1 and node2, do nothing
        pass


def color_mapped_nodes_wc(graph1, graph2, mapping2,graph3,mapping3,conflict_dict3):
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
    excluded_colors = [(1, 0, 0), (0, 0, 0), (1, 1, 1)]  #(red, black, and white)
    distinct_colors = [color for color in distinct_colors if color not in excluded_colors]

    
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
        try:
           new_graph2.edges[tar1, tar2]['color'] = colorc
        except:
           color_entire_walk(new_graph2, tar1, tar2, colorc)

    # Assign distinct colors to mapped nodes and their corresponding counterparts
    for idx, edge_mapping in mapping3.items():
        src1 = edge_mapping['src1']
        src2 = edge_mapping['src2']
        tar1 = edge_mapping['tar1']
        tar2 = edge_mapping['tar2']
        
        # Use a distinct color for this mapping
        
        if new_graph1.nodes[src1]['color'] is not 'white': ##NODE HAS BEEN COLORED COPY
           new_graph3.nodes[tar1]['color']  = new_graph1.nodes[src1]['color']
        else:
            color = distinct_colors.pop()  # Remove a color from the list
            new_graph1.nodes[src1]['color'] = color
            new_graph3.nodes[tar1]['color'] = color
        
        if new_graph1.nodes[src2]['color'] is not 'white':
            new_graph3.nodes[tar2]['color']  =new_graph1.nodes[src2]['color']
        else:     
            colord = distinct_colors.pop()  # Remove a color from the list
            # Assign the color to the corresponding nodes in both graphs
            new_graph1.nodes[src2]['color'] = colord
            new_graph3.nodes[tar2]['color'] = colord
        
        # Color the corresponding edges in both graphs
        if new_graph1.edges[src1, src2]['color']  is not 'black':
           try :
               new_graph3.edges[tar1, tar2]['color'] = new_graph1.edges[src1, src2]['color']
           except:
               color_entire_walk(new_graph3, tar1, tar2, new_graph1.edges[src1, src2]['color'])    
        else:
           colore = distinct_colors.pop()    
           new_graph1.edges[src1, src2]['color'] = colore
           
           try:
               new_graph3.edges[tar1, tar2]['color'] = colore
           except:
               color_entire_walk(new_graph3, tar1, tar2, colore)    
   
    ##-----------------------------------CONFLICT
    conside1=[]
    conside2=[]
    for item in conflict_dict3:
             if conflict_dict3[item]['loc_ver'] == False :
                new_graph1.nodes[item]['color'] = 'red'
                
                for itemc in conflict_dict3[item]['loc']:
                      if itemc!=item:  ### no self loops
                         try:
                            new_graph1.edges[item,itemc]['color']  = 'red'
                         except:
                            color_entire_walk(new_graph1, item,itemc,'red' )   
                m_node_a = conflict_dict3[item]['match']
                new_graph3.nodes[m_node_a]['color'] = 'red'
                
                for itemc in conflict_dict3[item]['loc_m']:
                    if itemc!=m_node_a:  ### no self loops
                      try:
                         new_graph3.edges[m_node_a,itemc]['color']  = 'red'
                      except:
                         color_entire_walk(new_graph3, m_node_a,itemc,'red' )        
                
                conside1 += ( conflict_dict3[item]['loc'] )
                conside2 += (conflict_dict3[item]['loc_m'] )
            
             if conflict_dict3[item]['date_ver'] == False :
                
                for itemc in conflict_dict3[item]['date'] :
                    if itemc!=item:  ### no self loops
                      try: 
                         new_graph1.edges[item,itemc]['color']  = 'red'
                      except:
                         color_entire_walk(new_graph1, item,itemc,'red' )  
                       
                m_node_a = conflict_dict3[item]['match']
                new_graph3.nodes[m_node_a]['color'] = 'red'
                
                for itemc in conflict_dict3[item]['date_m']:
                    if itemc!=m_node_a:  ### no self loops
                      try:
                        new_graph3.edges[m_node_a,itemc]['color']  = 'red'   
                      except:
                        color_entire_walk(new_graph3, m_node_a,itemc,'red' )    
                
                conside1+=( conflict_dict3[item]['date'] )
                conside2+= ( conflict_dict3[item]['date_m'] )       
          
        #pdb.set_trace()  
          
          
                    
        ##>
        ##conflict_dict[node_a]={'loc_ver':loc_verified,'loc':loc_neighbors_a,'loc_m':loc_m_neighbors_a,'date_ver':date_verified,'date':date_neighbors_a,'date_m':date_m_neighbors_a}  
    
    return new_graph1, new_graph2,new_graph3

#################???????????????????????????
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
        
        if new_graph1.nodes[src1]['color'] is not 'white': ##NODE HAS BEEN COLORED COPY
           new_graph3.nodes[tar1]['color']  = new_graph1.nodes[src1]['color']
        else:
            color = distinct_colors.pop()  # Remove a color from the list
            new_graph1.nodes[src1]['color'] = color
            new_graph3.nodes[tar1]['color'] = color
        
        if new_graph1.nodes[src2]['color'] is not 'white':
            new_graph3.nodes[tar2]['color']  =new_graph1.nodes[src2]['color']
        else:     
            colord = distinct_colors.pop()  # Remove a color from the list
            # Assign the color to the corresponding nodes in both graphs
            new_graph1.nodes[src2]['color'] = colord
            new_graph3.nodes[tar2]['color'] = colord
        
        # Color the corresponding edges in both graphs
        if new_graph1.edges[src1, src2]['color']  is not 'black':
           new_graph3.edges[tar1, tar2]['color'] = new_graph1.edges[src1, src2]['color']
        else:
           colore = distinct_colors.pop()    
           new_graph1.edges[src1, src2]['color'] = colore
           new_graph3.edges[tar1, tar2]['color'] = colore
     
    
    
    
    return new_graph1, new_graph2,new_graph3


#----------------------------------------------------------------------------------------------------------#


def get_search_terms(entity_relationship_graph):
    entities = get_ent(entity_relationship_graph)
    selected_terms = []
    for entity_type, entity_list in entities.items():
        if entity_list:
            selected_terms.append(random.choice(entity_list))
    search_term = " + ".join(selected_terms)
    return search_term

def get_search_terms_conditioned(entity_relationship_graph1, entity_relationship_graph2):
    g1_entities = get_ent(entity_relationship_graph1)
    g2_entities = get_ent(entity_relationship_graph2)
    
    com_term_events = construct_search_term(g1_entities['EVENT'], g2_entities['EVENT'])
    com_term_locations = construct_search_term(g1_entities['LOCATION'], g2_entities['LOCATION'])
    com_term_dates = construct_search_term(g1_entities['DATE'], g2_entities['DATE'])
    com_term_persons = construct_search_term(g1_entities['PERSON'], g2_entities['PERSON'])
    
    diff_term_events = construct_search_term_difference(g1_entities['EVENT'], g2_entities['EVENT'])
    diff_term_locations = construct_search_term_difference(g1_entities['LOCATION'], g2_entities['LOCATION'])
    diff_term_dates = construct_search_term_difference(g1_entities['DATE'], g2_entities['DATE'])
    diff_term_persons = construct_search_term_difference(g1_entities['PERSON'], g2_entities['PERSON'])
    
    common_terms = [com_term_events, com_term_locations, com_term_dates, com_term_persons]
    different_terms = [diff_term_events, diff_term_locations, diff_term_dates, diff_term_persons]
    
    search_terms = []
    for common_term, different_term in zip(common_terms, different_terms):
        if common_term:
            search_terms.append(random.choice(common_term))
        if different_term:
            search_terms.append(random.choice(different_term))
    
    search_term = " + ".join(search_terms)
    
    return search_term


def get_ent(entity_relationship_graph):
    entities = {
        'EVENT': [],
        'LOCATION': [],
        'DATE': [],
        'PERSON': []
    }
    
    for node in entity_relationship_graph.nodes(data=True):
        ent_type = node[1].get('ent_type')
        if ent_type in entities:
            entities[ent_type].append(node[0])
    
    return entities

def construct_search_term(graph1_entities, graph2_entities):
    common_entities = list(set(graph1_entities) & set(graph2_entities))
    search_terms = [entity for entity in common_entities if entity]
    return search_terms

def construct_search_term_difference(graph1_entities, graph2_entities):
    diff_entities = list(set(graph1_entities) - set(graph2_entities))
    search_terms = [entity for entity in diff_entities if entity]
    return search_terms


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

def print_graph_info(G):
    #pdb.set_trace()
    print("Nodes:")
    for node, data in G.nodes(data=True):
        print(f"Node {node}: {data}")
    print("\nEdges:")
    for edge in G.edges(data=True):
        print(edge)

def create_empty_image(width, height, background_color=(255, 255, 255)):
    # Create a new image with the specified width, height, and background color
    image = Image.new("RGB", (width, height), background_color)
    return image
    

def merge_edges_with_concatenation(graph1, graph2):
    # Create a new graph as the union of edges from both graphs
    union_graph = nx.compose(graph1, graph2)

    for edge in union_graph.edges:
        # Concatenate the action attribute if the edge occurs in both graphs
        if graph1.has_edge(*edge) and graph2.has_edge(*edge):
            action1 = graph1.edges[edge].get('action', '')
            action2 = graph2.edges[edge].get('action', '')
            combined_action = f"{action1} {action2}".strip()
            union_graph.edges[edge]['action'] = combined_action

    return union_graph
    

def save_graph_image(graph):
    plt.figure(figsize=(8, 8))
    node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]
    edge_colors = [graph.edges[edge]['color'] for edge in graph.edges()]
    nx.draw(graph, with_labels=True, font_weight='bold',font_size=35,node_color=node_colors, node_size=900, edge_color=edge_colors, width=15, cmap=plt.cm.tab10)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)
    graph_img = Image.open(buf)   
    plt.close()
    return graph_img 
        
def draw_graph_networkx(G):
    fig, ax = plt.subplots()
    pos = nx.spiral_layout(G)#nx.circular_layout(G)#nx.spring_layout(G) 
    nx.draw(G, pos, with_labels=False, font_weight='bold', ax=ax)
    node_labels = {node: f"{node}\n{attrs}" for node, attrs in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = {(source, target): attrs.get('action', '') for source, target, attrs in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    ax.set_axis_off()  
    
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    graph_img = Image.open(buf)   
    
    return graph_img 

def find_related_node(input_node,matched_nodes_tensor):
    matched_nodes_list = matched_nodes_tensor.tolist()
    for pair in matched_nodes_list:
        if pair[0] == input_node:
            return pair[1]

    return None




def cht_gen_graph_image(chtres):
    graph_gen_commands = chtres
    DG                 = nx.Graph()
    err                = 0
    try:
       exec(graph_gen_commands)
    except:
           #pdb.set_trace()
           print('bad command: <| '+ graph_gen_commands +' |>')     
           
           err+=1
           
    try:
        DG              = make_connected(DG)
        graph_img = draw_graph_networkx(DG)
    except:
        graph_img = empty_dg_img
    
    return graph_img,DG,err
    



def make_connected(graph):
    ifdum=0
    if nx.is_connected(graph):
        pass#print("Graph is already connected.")
    else:
        components = list(nx.connected_components(graph))
        if len(components) > 1:
            for i in range(1, len(components)):
                # Connect the components by adding an edge between nodes in different components
                node1 = min(components[0])
                node2 = min(components[i])
                graph.add_edge(node1, node2, action='<c>')

            #print("Graph is now connected.")

        else:
            pass
            #print("Graph is already connected.")

    ##adding selfloops
    #for node in graph.nodes():
    #    graph.add_edge(node, node,action='<s>')
        
    return graph



def valid_graph_text(chtres):
    graph_gen_commands = chtres
    DG                 = nx.Graph()
    err                = 0
    try:
       exec(graph_gen_commands)
       return True
    except:
       return False


def   get_graph_igc(claim_text):

          if claim_text=='no_text_evidence' or claim_text=='' or claim_text=='do_nothing':
               print('detected silly text <--> '+str(claim_text))
               return empty_dg_img,empty_dg,{}

          sucf ,er_graph          = checkupdate_claim(claim_text) #get_graphcode(claim_text)
          
          if sucf==0:
             code                 = er_to_graph_code(er_graph)
             timg,tdg,terr        = cht_gen_graph_image(code)
          else:
             timg,tdg             = [empty_dg_img,empty_dg ]
          return timg,tdg,er_graph 

def   get_graph_igc_conditioned(claim_text,cond_text,cond_er_graph):

          if claim_text=='no_text_evidence' or claim_text=='' or claim_text=='do_nothing':
               print('detected silly text <--> '+str(claim_text))
               return empty_dg_img,empty_dg,{}

          sucf ,er_graph          = checkupdate_evd(claim_text,cond_text,cond_er_graph) #get_graphcode(claim_text)
          
          if sucf==0:
             code                 = er_to_graph_code(er_graph)
             timg,tdg,terr        = cht_gen_graph_image(code)
          else:
             timg,tdg             = [empty_dg_img,empty_dg ]
          return timg,tdg,er_graph 
          
          
         
def remove_action_data(graph):
    for edge in graph['edges']:
        edge_data = edge[2]
        edge_data['action'] = ''
    return graph


def generate_conditioned_graph_rules(combined_rules,cond_er_graphx):
    cond_er_graphx     = remove_action_data(cond_er_graphx)
    cond_nodes         = cond_er_graphx['nodes']
    cond_masked_edges  = cond_er_graphx['edges']
    conditioned_rules = {}
    steps_sequence = [
        "Text Preprocessing Rule",
        "Entity Detection and Node Creation Rule ",
        "News Categorization",
        "Question Answering ",
        "Entity OR Node Filtering ",
        "Relationship Detection and Edge Creation Rule  ",
        "Relationship or Edge Filtering",
        "Overall guideline",
        "Verfication  Rule"
    ]

    # Generate conditioned rules based on the specified sequence
    for step in steps_sequence:
        if step in combined_rules:
            conditioned_rules[step] = combined_rules[step]
        else:
            if step=="Entity OR Node Filtering ":
               conditioned_rules[step]="Focus on extracting entities from the text that are either very similar or related to the entities in the reference entities list provided as <"+str(cond_nodes)+">. Please Standardize Entity Representations, by making sure that entities that are same or have similar meanings to the ones in the reference list are always represented consistently, in terms of name and description in this new Graph. Achieve this by copying the names and description from the reference entities list, when applicable. Remove the rest of the entities"
                
            if step=="Relationship or Edge Filtering": 
               conditioned_rules[step]="Understand that the expected output graph is resctricted to only have entities as nodes that are also present in the reference entities list. Thus focus on extracting the relationships between those entities which are also part of the reference list. These are the reference edges corresponding to the reference entities without the action indentified, <"+str(cond_masked_edges)+"> .  If the reference entities are part of the input text then you MUST predict the corresponding 'action' and add this edge to the output entity relationship graph "
                
    return conditioned_rules


def check_graph_dataformat_place_date(graph):
    nodes = graph.get('nodes', [])
    
    # Check entities of type location
    for node in nodes:
        if node[1]['ent_type'] == 'LOCATION':
            location_data = node[1]['data'].split(',')
            if len(location_data) != 3:
                return 1  # Return 1 if condition not met for location data
    
    # Check entities of type date
    for node in nodes:
        if node[1]['ent_type'] == 'DATE':
            date_data = node[1]['data'].split(',')
            if len(date_data) != 4:
                return 2  # Return 2 if condition not met for date data
    
    # Return 0 if both conditions are met
    return 0
    
def   get_ergraph_conditioned(claim_text,cond_er_graph):
      claim_text=claim_text.translate(str.maketrans('', '', string.punctuation))
      #graph_examples,selected_keys=generate_better_examples(4)
      #pdb.set_trace()
      cond_er_graphx     = cond_er_graph.copy()
      #cond_er_graphx     = remove_action_data(cond_er_graphx)
      #cond_nodes         = cond_er_graphx['nodes']
      #cond_masked_edges  = cond_er_graphx['edges']
      
      
      
      graph_rules_conditioned = generate_conditioned_graph_rules(graph_rules_combined,cond_er_graphx)
     
      try: 
         prompt_graph_const = f"You are a Journalist Assistant and NLP expert. Your task is to analyze a text story and generate an entity relationships graph representation that captures important entities and their relationships. Follow All the specified rules outlined in << '{graph_rules_conditioned}'>> to build this graph as a dictionary named er_dict. Focus on extracting entities from the text and ensure that the graph emphasizes the most relevant ones. Use the provided template {graph_format_template} for the output graph representation. Here's an example: {graph_format_ex}. A list of  examples is available ### '{graph_examples}' ### . Now for try this INPUT: '{claim_text}'. Ensure that the resulting graph retains only the most pertinent entities and relationships. Verify that all nodes are also present in the original text story. Provide the  graph as a dictionary adhering to the specified output format.The output er_dict should be interpretable as a Python dictionary when using ast.literal_eval(). Respond only with the output and avoid providing explanations."
         
         #prompt_graph_const = f"You are a Journalist Assistant and NLP expert. Your task is to analyze a text story and generate an entity relationships graph representation that captures important entities and their relationships. Follow All the specified rules outlined in << '{graph_rules_combined}'>> but focus on extracting entities from the text that are either very similar or related to the entities in the reference entities list provided as <{cond_nodes})>. Please Standardize Entity Representations, by making sure that entities that are same or have similar meanings to the ones in the reference list are always represented consistently, in terms of name and description in this new Graph. Achieve this by copying the names and description from the reference entities list, when applicable. Remove the rest of the entities. Understand that the expected output graph is resctricted to only have entities as nodes that are also present in the reference entities list. Thus focus on extracting the relationships between those entities which are also part of the reference list. These are the reference edges corresponding to the reference entities without the action indentified, <{cond_masked_edges}> .  If the reference entities are part of the input text then you MUST predict the corresponding 'action' and add this edge to the output entity relationship graph. Following the above guidelines build this entity relationships graph conditioned on the reference entities as a dictionary named er_dict. Use the provided template {graph_format_template} for the output graph representation. Here's an example: {graph_format_ex}. A list of  examples is available ### '{graph_examples}' ### . Now for try this INPUT: '{claim_text}'. Ensure that the resulting graph retains only the most pertinent entities and relationships. Verify that all  nodes of the graph are also present in the original text story.The output er_dict should be interpretable as a Python dictionary when using ast.literal_eval(). Respond only with the output and avoid providing explanations."
         
         
         #pdb.set_trace()
         res_graph_stuff = get_response_ch_api_wrap(prompt_graph_const)
         #pdb.set_trace()
         er_graph        = ast.literal_eval(res_graph_stuff.strip())
         flag    ,graph  = create_graph_from_output(er_graph)

         code            = er_to_graph_code(er_graph)
 
         if check_graph_dataformat_place_date(er_graph)!=0:
            print("graph had issues in date and place format, rejected")
            raise ValueError("graph had issues in date and place format, rejected")
                   
         if (flag==0)and valid_graph_text(code):
             print('upvote off')#upvote_sample(selected_keys, 0.001)
             return 0,er_graph
         else:
             #pdb.set_trace()
             return 2,er_graph      
      except:
         print('< graph build failed > <retry with better examples>')
         #pdb.set_trace()
         return 1, 'no ergraph' 
 




def   get_ergraph(claim_text):
      claim_text=claim_text.translate(str.maketrans('', '', string.punctuation))
      
      #pdb.set_trace()
      
      
      #graph_examples,selected_keys=generate_better_examples(4)
      try: 
         prompt_graph_const = f"You are a Journalist Assistant, and NLP expert. Your task is to look at a text story and generate a entity relationships graph representation that captures all entities and relationships. Follow all the rules defined in the <<{graph_rules_combined}>> to build this graph as a dictionary called er_dict. The output format must adhere to the specified template: {graph_format_template} and here is an example: {graph_format_ex}. A list of graph examples is provided ### '{graph_examples}'###.  Now for try this INPUT: '"+claim_text+"'. The output er_dict should be interpretable as  python dictionary when I apply ast.literal_eval(output). Only respond with the output  AND ABSOLUTELY NOTHING ELSE. Do not provide explanations."
         
         res_graph_stuff = get_response_ch_api_wrap(prompt_graph_const)
         er_graph        = ast.literal_eval(res_graph_stuff.strip())
         flag    ,graph  = create_graph_from_output(er_graph)
         code            = er_to_graph_code(er_graph)
         #pdb.set_trace()
         if check_graph_dataformat_place_date(er_graph)!=0:
            raise ValueError("graph had issues in date and place format, rejected")
         
         if (flag==0)and valid_graph_text(code):
             print('upvote off')#upvote_sample(selected_keys, 0.001)
             return 0,er_graph
         else:
             #pdb.set_trace()
             return 2,er_graph      
      except:
         print('< graph build failed > <retry with better examples>')
         return 1, 'no ergraph'     
 
 
 
 
 
       
         
def checkupdate_claim(text):
    global chatgpt_res_dict
    flag=1
    org_text=text
    if  text in chatgpt_res_dict: ##dinero
            print('loading saved claim er')
            er_graph = chatgpt_res_dict[text]
            flag=0
    else:
            num_tries=0
            er_graph=empty_er_graph
            while(num_tries<5):
                   flag,er_graph  =  get_ergraph(text)
                   if    flag==0: 
                          break
                   print('bad graph claim retrying')  
                   #pdb.set_trace()
                   num_tries+=1   
            chatgpt_res_dict[org_text]            =   er_graph  
            np.savez(dictfile,chatgpt_res_dict=chatgpt_res_dict)


    return flag,er_graph 

def checkupdate_evd(text,cond_text,cond_er_graph):
    global chatgpt_res_dict
    flag=1
    text_key= text +'<c>'+ cond_text
    
    if text_key in chatgpt_res_dict: ##dinero
            print('loading saved evidence er') 
            er_graph = chatgpt_res_dict[text_key]
            flag=0
    else:
            #pdb.set_trace()
            num_tries=0
            er_graph=empty_er_graph
            while(num_tries<5):
                   flag,er_graph  =  get_ergraph_conditioned(text,cond_er_graph)
                   if    flag==0: 
                          break
                   print('bad graph evidence retrying')  
                   #pdb.set_trace()
                   num_tries+=1   
            chatgpt_res_dict[text_key]            =   er_graph   #### text+claim
            np.savez(dictfile,chatgpt_res_dict=chatgpt_res_dict)


    return flag,er_graph    

def cleancheckupdate(text):
    if text=='no_text_evidence':
       return text
    
    global clean_chatgpt_res_dict
    
    if text in clean_chatgpt_res_dict:
       value = clean_chatgpt_res_dict[text]
    else:
       response_value                                = get_response_ch_api_wrap(cprompt +" "+ text)
       
      
       #response_value    = ast.literal_eval(response_value )
       value             = response_value #['summary']

       
       clean_chatgpt_res_dict[text]                  = value  
       np.savez(clean_dictfile,clean_chatgpt_res_dict=clean_chatgpt_res_dict)


    return value 
        
def del_build_graph(text):
    #pdb.set_trace()
    if text=='no_text_evidence' or text=='' or text=='do_nothing':
          print('detected silly text <--> '+str(text))
          return empty_dg_img,empty_dg,text,99
    num_tries=0
    flag,er_graph   = checkupdate(text)  
    code            = er_to_graph_code(er_graph)
    if flag==0:
         gimg,dg,err     = cht_gen_graph_image(code) 
    else:
       return empty_dg_img,empty_dg,text,99    
 

    return gimg,dg,code,err



def encode_text(text, tokenizer, bertmodel):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bertmodel(**inputs)
    return outputs.last_hidden_state.mean(dim=1)




def calculate_node_similarity(node_embeddings1, node_embeddings2):
    node_similarities = torch.from_numpy( pw_cosine_similarity(node_embeddings1, node_embeddings2))

    return node_similarities 


def find_closest_node(node, all_nodes, matched_nodes):
    
    try:
      closest_node=all_nodes[np.nonzero( matched_nodes[ all_nodes.index(node),: ] ).item()]
    except:
      closest_node=None

    return closest_node


def find_path_with_actions(graph, node1, node2):
    try:
        path = nx.shortest_path(graph, source=node1, target=node2)
        actions = [graph[path[i]][path[i+1]]['action'] for i in range(len(path)-1)]
        
        if len(path)>0:
            #path_with_actions = [f"{path[i]} {actions[i]}" for i in range(len(path)-1)] + [path[-1]]
            
            path_with_actions = [word for i in range(len(path)-1) for word in [path[i], actions[i]]] + [path[-1]]
            
            # Remove the start and end nodes from the path
            path_with_actions = ' '.join(path_with_actions[1:-1])
            return path_with_actions
            
        
        else :
            return 1  
        
    except nx.NetworkXNoPath:
        print('all connected how here?')
        #pdb.set_trace()
        return 1



def examined(ransamidx):
    folder_path = DATA_DIR+res_folder+str(ransamidx)  
    if os.path.exists(folder_path):
          return True
    else:

          return False     
          
def load_xvd_scores(ransamidx):
    output_folder = DATA_DIR+res_folder+str(ransamidx) 
    data=np.load(output_folder+'/vis_es.npz',allow_pickle=True)
    return data[data.files[0]].tolist(),data[data.files[1]].tolist(),data[data.files[2]].tolist(),data[data.files[3]].tolist()                 
def save_xvd_scores(ransamidx,vs,vs_l,vts,ts):
    output_folder = DATA_DIR+res_folder+str(ransamidx)
    os.makedirs(output_folder, exist_ok=True)
    np.savez(output_folder+'/vis_es.npz',vs=vs,vs_l=vs_l,vts=vts,ts=ts)


def load_outputs_from_folder(ransamidx):
    folder_path = DATA_DIR+res_folder+str(ransamidx)    
    
    with open(os.path.join(folder_path, 'cl_edg_count.txt'), 'r') as f:
        cl_edg_count = int(f.read())

    with open(os.path.join(folder_path, 'qafrac.txt'), 'r') as f:
        qafrac = float(f.read())#qafrac = int(f.read())

    with open(os.path.join(folder_path, 'exps.txt'), 'r') as f:
        exps = f.read()
   
    with open(os.path.join(folder_path, 'vis_sim.txt'), 'r') as f:
        vis_sim = float(f.read())

    with open(os.path.join(folder_path, 'bin_pred.txt'), 'r') as f:
        bin_pred = float(f.read())


    with open(os.path.join(folder_path, 'evidence_vtxt.txt'), 'r') as f:
        evidence_vtxt = f.read()
    with open(os.path.join(folder_path, 'evidence_ttxt.txt'), 'r') as f:
        evidence_ttxt = f.read()                

    vsim_list = []
    with open(os.path.join(folder_path, 'vsim_list.txt'), 'r') as f:
         lines = [line.strip() for line in f]
    vsim_list  = [float(line.strip()) for line in lines[:-1]] 
    vsim_list.append([float(value) for value in lines[-1].strip('[]').split()] )


    ####
    #runtime eval logic update
    ####
            
    claim_graph_img     = Image.open(os.path.join(folder_path, 'claim_graph_img.jpg'))
    claim_graph_img_ANN = Image.open(os.path.join(folder_path, 'claim_graph_img_ANN.jpg'))
    vevd_graph_img_ANN      = Image.open(os.path.join(folder_path, 'vevd_graph_img_ANN.jpg'))
    tevd_graph_img_ANN      = Image.open(os.path.join(folder_path, 'tevd_graph_img_ANN.jpg'))

    
   
    return bin_pred,cl_edg_count,qafrac, exps, claim_graph_img, claim_graph_img_ANN, vevd_graph_img_ANN, tevd_graph_img_ANN, vis_sim,vsim_list,evidence_vtxt, evidence_ttxt


def save_outputs(bin_pred,ransamidx,  cl_edg_count, qafrac, exps, img_p,claim_graph_img, claim_graph_img_ANN, evd_grid,evidence_v_p,vevd_graph_img, vevd_graph_img_ANN, tevd_graph_img, tevd_graph_img_ANN, vis_sim,vsim_list,evidence_vtxt,evidence_ttxt):
    output_folder =res_folder+str(ransamidx)
    os.makedirs(output_folder, exist_ok=True)

    # Save outputs to the folder
    
    
    
    
    with open(os.path.join(output_folder, 'cl_edg_count.txt'), 'w') as f:
        f.write(str(cl_edg_count))

    with open(os.path.join(output_folder, 'qafrac.txt'), 'w') as f:
        f.write(str(qafrac))

    with open(os.path.join(output_folder, 'exps.txt'), 'w') as f:
        f.write(exps)

    with open(os.path.join(output_folder, 'vis_sim.txt'), 'w') as f:
        f.write(str(vis_sim))
   
    with open(os.path.join(output_folder, 'bin_pred.txt'), 'w') as f:
        f.write(str(bin_pred))     

    with open(os.path.join(output_folder, 'vsim_list.txt'), 'w') as f:
        for score in vsim_list:
            f.write(str(score) + '\n')
    
    
    #help here #claim_img load from path img_p
    
    
    
    claim_img = Image.open(img_p)
    claim_img.save(os.path.join(output_folder, 'claim_image.jpg'))
    


    
    claim_graph_img.save(os.path.join(output_folder, 'claim_graph_img.jpg'))
    claim_graph_img_ANN.save(os.path.join(output_folder, 'graph_claim.jpg'))


    vevd_img = Image.open(evidence_v_p)
    vevd_img.save(os.path.join(output_folder, 'evidence_image.jpg'))

    evd_grid.save(os.path.join(output_folder, 'visual_evidences.jpg')) 
    vevd_graph_img.save(os.path.join(output_folder, 'vevd_graph_img.jpg'))
    vevd_graph_img_ANN.save(os.path.join(output_folder, 'graph_evidence_vis.jpg'))
    
    tevd_graph_img.save(os.path.join(output_folder, 'tevd_graph_img.jpg'))
    tevd_graph_img_ANN.save(os.path.join(output_folder, 'graph_evidence_text.jpg'))
    
    #pdb.set_trace()  
    with open(os.path.join(output_folder, 'evidence_vtxt.txt'), 'w') as f:
        f.write(evidence_vtxt)
    with open(os.path.join(output_folder, 'evidence_ttxt.txt'), 'w') as f:
        f.write(evidence_ttxt)
        
        

    jdict={}
    jdict[ransamidx]={'claim_edge_count':cl_edg_count,'claim_edge_verified_fraction':qafrac,'exps':exps,'visual_similarity':vis_sim,'bin_prediction':bin_pred,'evidence_vtxt':evidence_vtxt,'evidence_ttxt':evidence_ttxt}  
    with open(os.path.join(output_folder, 'jdict.json'), 'w') as json_file:
             json.dump(jdict, json_file)
    #pdb.set_trace()         

##>>
####
   


def zsal(ransamidx,    img_p,       caps, evd_grid,evidence_v_p, evidence_vtxt, evidence_ttxt, vis_sim_thres, node_similarity_threshold,  edge_similarity_threshold, evidence_v_scr,evidence_v_l_scr,evidence_t_scr):
    
    print('building graphs-------------')
    #pdb.set_trace()
    claim_graph_img, CDG, cres     = get_graph_igc( caps)
    tcres=copy.deepcopy(cres)
    
    ###conditional or not
    ##pdb.set_trace()
    vevd_graph_img,  VDG, vres     = get_graph_igc_conditioned( evidence_vtxt ,caps, tcres)
    tevd_graph_img,  TDG, tres     = get_graph_igc_conditioned( evidence_ttxt ,caps, tcres)
    
    #pdb.set_trace()
    #vevd_graph_img,  VDG, vres     = get_graph_igc( evidence_vtxt)
    #tevd_graph_img,  TDG, tres     = get_graph_igc( evidence_ttxt)
    print('building graphs------------- <done>')
    #pdb.set_trace()
    
    try:
       print('vistxt graph')
       SV,EVG, con_V,sup_V,STV,vmaps,vconflict_dict=graph_similarity(CDG.copy(),caps,VDG.copy(),evidence_vtxt,node_similarity_threshold, edge_similarity_threshold,' <vis> ')
       print('text graph')
       ST,ETG, con_T,sup_T,STT,tmaps,tconflict_dict=graph_similarity(CDG.copy(),caps,TDG.copy(),evidence_ttxt,node_similarity_threshold, edge_similarity_threshold,' <txt> ')
    except:
       print('similariy failed !!!')
       pdb.set_trace()
    print('graph similarity checks------------- <done>')
    try:
        vis_sim,vis_sim_list=evidence_v_scr,evidence_v_l_scr
    except:
        print('image data issue: ')
        print(evidence_v_p)
        print(img_p) 
        vis_sim=0
        vis_sim_list=[]
    print('visual similarity checks------------- <done>')

 
 
    #######################################################################
    qafrac=0
    bin_pred=1
    exps  =' '
    cl_edg_count=max(1,CDG.number_of_edges())
    print('<') 
    print(ST)
    print('---')
    print(evidence_t_scr)
    print(cl_edg_count)
    if len(con_T)==0:
        ST=(ST/cl_edg_count)
    else:
        print('text conflict < ')
        print(con_T)
        ST=0
    if len(con_V)==0:
       SV=(SV/cl_edg_count)
    else:
       print('image conflict < ')
       print(con_V)
											  
       SV=0
     
    ###<tdx> do we need image conflict checks?
    print(SV)      
    print('---')
    print(vis_sim_list)
    print(vis_sim)
    print('>')
					
    topk=4
    vis_vpfocs=vis_sim_list[0:6]
    vscr_vit,vscr_place,vscr_face,vscr_obj,vscr_cap,vscr_sct=vis_vpfocs
    score_cat=['vit','place','faces','objects','caption','scene_text']
    vis_cuml, score_cat_indices = sum(sorted(vis_vpfocs)[-topk:]) / topk, [score_cat[i] for i in sorted(range(len(vis_vpfocs)), key=lambda i: vis_vpfocs[i], reverse=True)[:topk]]

    
    if dataset_name=='remiss':  
       abl_file='remiss_data/bcn19/ablation.npz'  
       
    if dataset_name=='ccnd':    
       abl_file='newsclip_data/ablation.npz'   

    all_res  =np.load(abl_file,allow_pickle=True)
    all_res  =all_res[all_res.files[0]]
    all_res  =all_res.flat[0]
    
    #all_res={}
    
    abl_sim_txt,abl_sim_vxv,  abl_gm_txt,abl_gm_txvt, abl_im_vxv, bin_pred=1,1,1,1,1,1
    xt=False
    xv=False
    texps='  '
    vexps='  '
    if evidence_ttxt!='no_text_evidence':
        xt=True
        texps='XT(V) Rejects'
        if evidence_t_scr  > txt_sim_thres:
           abl_sim_txt=0
           texps     +=  ' XT(V) sim verifies  '
        if (evidence_t_scr  > txt_sim_thres-0.04) and ( (ST > 0.3) or ( len(con_T)==0 and len(sup_T) > 1  ) ):
           abl_gm_txt     =0 
           texps     += '  XT(V) Graph verifies  '       
        if len(con_T)>0:
          texps += '+ XT(V) conflicts in location / date '
          abl_gm_txt     =1

    if  vscr_vit       > vis_sim_thres: 
        abl_sim_vxv=0  
        vexps     +=  ' XV(T) sim(vit) verifies  '  
    if  vis_cuml       > vis_sim_thres: 
        abl_im_vxv     =0 
        vexps     +=  ' XV(T) sim(vpfogs) verifies  '   
    if  ( (vis_cuml       > vis_sim_thres - 0.04) and (SV > 0.45)   ) or ( (vis_cuml > vis_sim_thres ) and ( SV > 0.1) )  : 
        abl_gm_txvt     =0     
        vexps     +=  ' XV(T) GM verifies  '   
    exps = texps +' - ' +vexps
    bin_pred=1
    if  (xt==True):
       bin_pred =(abl_gm_txt)
    bin_pred= bin_pred  and  (( abl_im_vxv ) and (abl_gm_txvt))
    
    print(exps)
    #pdb.set_trace()
    all_res[ransamidx]={'xt':xt,'sim_txt':abl_sim_txt,'gmatch_txt':abl_gm_txt,'sim_vxv':abl_sim_vxv,'imatch_vxv':abl_im_vxv, 'gmatch_txvt':abl_gm_txvt, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
    
    np.savez(abl_file,all_res=all_res)
    
    
    claim_graph_img_ANN=create_empty_image(200,200)     
    EG=merge_edges_with_concatenation(ETG, EVG)
    cl_edg_count=CDG.number_of_edges()
    ev_edg_count=EG.number_of_edges()
    if ev_edg_count> 0 and  cl_edg_count >0: 
        qafrac      =ev_edg_count/cl_edg_count
    else:
        qafrac      =0
    try:
      claim_graph_img_ANN=draw_graph_networkx(EG)
    except:
      claim_graph_img_ANN=create_empty_image(200,200) 
    

    new_CDG, new_VDG ,new_TDG= color_mapped_nodes_wc(CDG, VDG, vmaps,TDG,tmaps,tconflict_dict)
    claim_graph_img_ANN=save_graph_image(new_CDG)
    vevd_graph_img_ANN =save_graph_image(new_VDG)
    tevd_graph_img_ANN =save_graph_image(new_TDG)
    #pdb.set_trace()
    
    save_outputs(bin_pred,ransamidx, cl_edg_count, qafrac, exps,img_p, claim_graph_img, claim_graph_img_ANN, evd_grid,evidence_v_p,vevd_graph_img,vevd_graph_img_ANN, tevd_graph_img,tevd_graph_img_ANN, vis_sim,vis_sim_list,evidence_vtxt,evidence_ttxt)
    
    print('zsal_success----------------<>')
    return bin_pred,cl_edg_count,qafrac,exps,claim_graph_img,claim_graph_img_ANN,vevd_graph_img_ANN,tevd_graph_img_ANN,vis_sim ,vis_sim_list 
####################






def zsal_text(ransamidx,    img_p,       caps, evd_grid,evidence_v_p, evidence_vtxt, evidence_ttxt, vis_sim_thres, node_similarity_threshold,  edge_similarity_threshold, evidence_v_scr,evidence_v_l_scr,evidence_t_scr):
    
    print('building graphs-------------')
    claim_graph_img, CDG, cres     = get_graph_igc( caps)
    tcres=copy.deepcopy(cres)
    vevd_graph_img,  VDG, vres     = get_graph_igc_conditioned( evidence_vtxt ,caps, tcres)
    
    #tevd_graph_img,  TDG, tres     = get_graph_igc_conditioned( evidence_ttxt ,caps, tcres)
    tevd_graph_img=create_empty_image(200,200) 
    
    
    print('building graphs------------- <done>')

    
    try:
       print('vistxt graph')
       SV,EVG, con_V,sup_V,STV,vmaps,vconflict_dict=graph_similarity(CDG.copy(),caps,VDG.copy(),evidence_vtxt,node_similarity_threshold, edge_similarity_threshold,' <vis> ')
       print('text graph')
       #ST,ETG, con_T,sup_T,STT,tmaps,tconflict_dict=graph_similarity(CDG.copy(),caps,TDG.copy(),evidence_ttxt,node_similarity_threshold, edge_similarity_threshold,' <txt> ')
    except:
       print('similariy failed !!!')

    print('graph similarity checks------------- <done>')
    try:
        vis_sim,vis_sim_list=evidence_v_scr,evidence_v_l_scr
    except:
        print('image data issue: ')
        print(evidence_v_p)
        print(img_p) 
        vis_sim=0
        vis_sim_list=[]
    print('visual similarity checks------------- <done>')

 
 
    #######################################################################
    qafrac=0
    bin_pred=1
    exps  =' '
    cl_edg_count=max(1,CDG.number_of_edges())
    print('<') 
    #print(ST)
    #print('---')
    #print(evidence_t_scr)
    #print(cl_edg_count)
    #if len(con_T)==0:
    #    ST=(ST/cl_edg_count)
    # else:
    #    print('text conflict < ')
    #   print(con_T)
    #    ST=0
    if len(con_V)==0:
       SV=(SV/cl_edg_count)
    else:
       print('image conflict < ')
       print(con_V)
											  
       SV=0
     
    ###<tdx> do we need image conflict checks?
    print(SV)      
    print('---')
    print(vis_sim_list)
    print(vis_sim)
    print('>')
					
    topk=4
    vis_vpfocs=vis_sim_list[0:6]
    vscr_vit,vscr_place,vscr_face,vscr_obj,vscr_cap,vscr_sct=vis_vpfocs
    score_cat=['vit','place','faces','objects','caption','scene_text']
    vis_cuml, score_cat_indices = sum(sorted(vis_vpfocs)[-topk:]) / topk, [score_cat[i] for i in sorted(range(len(vis_vpfocs)), key=lambda i: vis_vpfocs[i], reverse=True)[:topk]]

    
    '''if dataset_name=='remiss':  
       abl_file='remiss_data/bcn19/ablation.npz'  
       
    if dataset_name=='ccnd':    
       abl_file='newsclip_data/ablation.npz'   

    all_res  =np.load(abl_file,allow_pickle=True)
    all_res  =all_res[all_res.files[0]]
    all_res  =all_res.flat[0]'''
    
    #all_res={}
    
    abl_sim_txt,abl_sim_vxv,  abl_gm_txt,abl_gm_txvt, abl_im_vxv, bin_pred=1,1,1,1,1,1
    xt=False
    xv=False
    texps='  '
    vexps='  '
    if evidence_ttxt!='no_text_evidence':
        xt=True
        texps='XT(V) Rejects'
        if evidence_t_scr  > txt_sim_thres:
           abl_sim_txt=0
           texps     +=  ' XT(V) sim verifies  '
        if (evidence_t_scr  > txt_sim_thres-0.04) and ( (ST > 0.3) or ( len(con_T)==0 and len(sup_T) > 1  ) ):
           abl_gm_txt     =0 
           texps     += '  XT(V) Graph verifies  '       
        if len(con_T)>0:
          texps += '+ XT(V) conflicts in location / date '
          abl_gm_txt     =1

    if  vscr_vit       > vis_sim_thres: 
        abl_sim_vxv=0  
        vexps     +=  ' XV(T) sim(vit) verifies  '  
    if  vis_cuml       > vis_sim_thres: 
        abl_im_vxv     =0 
        vexps     +=  ' XV(T) sim(vpfogs) verifies  '   
    if  ( (vis_cuml       > vis_sim_thres - 0.04) and (SV > 0.45)   ) or ( (vis_cuml > vis_sim_thres ) and ( SV > 0.1) )  : 
        abl_gm_txvt     =0     
        vexps     +=  ' XV(T) GM verifies  '   
    exps = texps +' - ' +vexps
    bin_pred=1
    if  (xt==True):
       bin_pred =(abl_gm_txt)
    bin_pred= bin_pred  and  (( abl_im_vxv ) and (abl_gm_txvt))
    
    print(exps)
    #pdb.set_trace()
    #all_res[ransamidx]={'xt':xt,'sim_txt':abl_sim_txt,'gmatch_txt':abl_gm_txt,'sim_vxv':abl_sim_vxv,'imatch_vxv':abl_im_vxv, 'gmatch_txvt':abl_gm_txvt, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
    #np.savez(abl_file,all_res=all_res)
    
    
    claim_graph_img_ANN=create_empty_image(200,200)     
    print('NO TEXT GRAPH CONCAT')
    EG= EVG#merge_edges_with_concatenation(ETG, EVG)
    cl_edg_count=CDG.number_of_edges()
    ev_edg_count=EG.number_of_edges()
    if ev_edg_count> 0 and  cl_edg_count >0: 
        qafrac      =ev_edg_count/cl_edg_count
    else:
        qafrac      =0
    try:
      claim_graph_img_ANN=draw_graph_networkx(EG)
    except:
      claim_graph_img_ANN=create_empty_image(200,200) 
    
    print('NO TEXT GRAPH CONCAT')
    new_CDG, new_VDG ,new_TDG= color_mapped_nodes_wc(CDG, VDG, vmaps,VDG,vmaps,vconflict_dict)
    claim_graph_img_ANN=save_graph_image(new_CDG)
    vevd_graph_img_ANN =save_graph_image(new_VDG)
    tevd_graph_img_ANN =save_graph_image(new_TDG)
    
    #pdb.set_trace()
    
    save_outputs(bin_pred,ransamidx, cl_edg_count, qafrac, exps,img_p, claim_graph_img, claim_graph_img_ANN, evd_grid,evidence_v_p,vevd_graph_img,vevd_graph_img_ANN, tevd_graph_img,tevd_graph_img_ANN, vis_sim,vis_sim_list,evidence_vtxt,evidence_ttxt)
    
    print('zsal_success----------------<>')
    return STV,claim_graph_img_ANN, vevd_graph_img_ANN,SV
   
   
    #return bin_pred,cl_edg_count,qafrac,exps,claim_graph_img,claim_graph_img_ANN,vevd_graph_img_ANN,tevd_graph_img_ANN,vis_sim ,vis_sim_list 
    #return ser_terms, new_CDG, new_VDG, vtx_va
   



####################

def is_location(node):
    # Assuming 'ent_type' is a key in the node attributes dictionary
    return node.get('ent_type') == 'LOCATION'

def is_date(node):
    # Assuming 'ent_type' is a key in the node attributes dictionary
    return node.get('ent_type') == 'DATE'


def clean_unknown(input_list):
    # Define a set of variations of 'unknown' strings
    unknown_variations = {'unk', 'unknown', 'Unknown',''}
    
    # Return a list comprehension excluding elements that match the variations
    return [item for item in input_list if item.lower() not in unknown_variations]
    
def get_data(nodes,graph):
       voc=[]
       for node in nodes:
           item_data  = graph.nodes[node]['data']
           split_data = re.split(',| ', item_data) + re.split(',| ', node)
           voc=voc+split_data
       return voc 

def is_valid_day(day):
    try:
        day = int(day)
        return 1 <= day <= 31
    except ValueError:
        return False

def is_valid_month(month):
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    return month in months

def is_valid_year(year):
    try:
        year = int(year)
        return year >= 0
    except ValueError:
        return False

def date_conflict(date1, date2):
    # Splitting dates into day of the week, day of the month, month, and year
    day_of_week1, day_of_month1, month1, year1 = [part.strip().lower() for part in date1.split(',')]
    day_of_week2, day_of_month2, month2, year2 = [part.strip().lower() for part in date2.split(',')]


    
    
    # Checking if each part is valid, if not, treat as 'unk'
    day_of_week1 = day_of_week1 if is_valid_day(day_of_week1) else 'unk'
    day_of_month1 = day_of_month1 if is_valid_day(day_of_month1) else 'unk'
    month1 = month1 if is_valid_month(month1) else 'unk'
    year1 = year1 if is_valid_year(year1) else 'unk'
    
    day_of_week2 = day_of_week2 if is_valid_day(day_of_week2) else 'unk'
    day_of_month2 = day_of_month2 if is_valid_day(day_of_month2) else 'unk'
    month2 = month2 if is_valid_month(month2) else 'unk'
    year2 = year2 if is_valid_year(year2) else 'unk'
    
    # Ignoring unknown values (unk)
    day_of_week_conflict = day_of_week1 != 'unk' and day_of_week2 != 'unk' and day_of_week1 != day_of_week2
    day_of_month_conflict = day_of_month1 != 'unk' and day_of_month2 != 'unk' and day_of_month1 != day_of_month2
    month_conflict = month1 != 'unk' and month2 != 'unk' and month1 != month2
    year_conflict = year1 != 'unk' and year2 != 'unk' and year1 != year2
    
    # Checking if conflicts exist
    if (day_of_week_conflict or day_of_month_conflict or month_conflict or year_conflict):
        return True
    else:
        return False
        
def check_null_address(address):
    # Split the address by commas
    parts = address.split(', ')
    
    # Check if there are exactly three parts and if each part is 'unk'
    return len(parts) == 3 and all(part == 'unk' for part in parts)
    
    
def address_conflict(address1, address2): ##tds newyork city to new york,or   just set after country , city and state is often confused
    # Splitting addresses into city, state, and country
    city1, state1, country1 = [part.strip().lower() for part in address1.split(',')]
    city2, state2, country2 = [part.strip().lower() for part in address2.split(',')]
    
    # Ignoring unknown values (unk)
    ## fix city1,unk,con   unk,unk,unk is 71 tds
    city_conflict  = city1  != 'unk' and  city2 != 'unk' and  city1  not in [city2,state2]    # city1 != city2       ##
    state_conflict = state1 != 'unk' and state2 != 'unk' and  state1 not in [city2,state2]  #state1 != state2  ## 
    #
    country_conflict = country1 != 'unk' and country2 != 'unk' and country1 != country2
    
    city_state_conflict=city_conflict or state_conflict
     
     
     
    # Checking if conflicts exist and address are not null 
    if city_state_conflict or country_conflict and (not(check_null_address(address1)) and not(check_null_address(address1))):
        return True
    else:
        return False

def check_items_match(graph1, graph2, items_a, items_b):#, matched_nodes, all_nodes):
   
    #match_based_score_map= any(find_closest_node(items_a[i], all_nodes, matched_nodes) == items_b[x] for i in range(len(items_a)) for x in range(len(items_b)))

    match_based_score_data=True #tds2 city,unk,unk  unkunkunk
    
    
    #pdb.set_trace
    #check for date_conflict
    if len(items_a)>0 and len(items_b)>0 and is_date(graph1.nodes[items_a[0]]):   
            match_based_score_data = any(not(date_conflict(graph1.nodes[a]['data'], graph2.nodes[b]['data'])) for a in items_a for b in items_b)
    
    #check for address_conflict
    if len(items_a)>0 and len(items_b)>0 and is_location(graph1.nodes[items_a[0]]):  
           match_based_score_data = any(not(address_conflict(graph1.nodes[a]['data'], graph2.nodes[b]['data'])) for a in items_a for b in items_b)
    
    if  match_based_score_data==True:
           print('-----------------------support found ')
    else:
           print('----node data doesnt help-----------')     
    
    
    return match_based_score_data  # match_based_score_map or 
    
    
    




    
    

def get_neighbors_up_to_depth(graph, node, depth):
    """
    Returns all neighbors of a node up to a certain depth in the graph.
    """
    ego_graph = nx.ego_graph(graph, node, radius=depth)
    return list(ego_graph.nodes())



def graph_neighbor_match(graph1, neighbors_a, graph2, m_neighbors_a):
        # entities from neighbor_a that are about date and location type
        loc_neighbors_a = [n for n in neighbors_a if is_location(graph1.nodes[n])]
        date_neighbors_a = [n for n in neighbors_a if is_date(graph1.nodes[n])]
        
        # entities from m_neighbor_a that are about date and location type
        loc_m_neighbors_a = [n for n in m_neighbors_a if is_location(graph2.nodes[n])]
        date_m_neighbors_a = [n for n in m_neighbors_a if is_date(graph2.nodes[n])]

        # Check if any location items can be matched
        
        #pdb.set_trace()
        loc_verified = check_items_match(graph1, graph2, loc_neighbors_a, loc_m_neighbors_a)#, matched_nodes, all_nodes)
      
        
        # Check if any date items can be matched
        date_verified =check_items_match(graph1, graph2, date_neighbors_a, date_m_neighbors_a)#, matched_nodes, all_nodes)
       
        return loc_verified, date_verified, loc_neighbors_a, loc_m_neighbors_a, date_neighbors_a, date_m_neighbors_a

def check_all_neighbor_conflicts(graph1, graph2, matched_nodes, all_nodes):
    conflicts = []
    supports  = []
    conflict_dict={}
    #
    #
    node_mapped_AND_checked=False
    for node_a in graph1.nodes():
        m_node_a = find_closest_node(node_a, all_nodes, matched_nodes)
        print('<--')
        print(node_a)
        print(m_node_a)
        print('--->')
        
        if not m_node_a:
            print('-------------<Not matched>')
            continue
        print('-------------<matched>') 
        node_mapped_AND_checked=True 
        # matched node found, check neighbors    	     
        neighbors_a   = get_neighbors_up_to_depth(graph1, node_a, 7) 
        m_neighbors_a = get_neighbors_up_to_depth(graph2, m_node_a, 7)
        #pdb.set_trace()
        loc_verified, date_verified, loc_neighbors_a, loc_m_neighbors_a, date_neighbors_a, date_m_neighbors_a= graph_neighbor_match(graph1, neighbors_a, graph2, m_neighbors_a)

        print ('<-------------------------------------------loc 1') 
        print (loc_neighbors_a)
        print ('--------------------------------------------loc 2') 
        print (loc_m_neighbors_a)
        print ('--------------------------------------------date 1') 
        print (date_neighbors_a)
        print ('--------------------------------------------date 2') 
        print (date_m_neighbors_a)
        print ('-------------------------------------------------->') 
        #pdb.set_trace()
        print ('---------- support or conflict ----------')
        if loc_verified and date_verified: 
            print ('support found')
            supports.append(node_a)
        else:
            print ('conflict detected')
            #pdb.set_trace()
            conflicts.append(node_a) 
            conflict_dict[node_a]={'match':m_node_a,'loc_ver':loc_verified,'loc':loc_neighbors_a,'loc_m':loc_m_neighbors_a,'date_ver':date_verified,'date':date_neighbors_a,'date_m':date_m_neighbors_a}

            
            
            
            

    if node_mapped_AND_checked==False:
        print('no matching nodes, <--------checking overall--------->')
        ## if the two graphs have no node matching, check for conflict 
        node_a        = max(graph1.nodes(), key=graph1.degree)
        m_node_a      = max(graph2.nodes(), key=graph2.degree)
        neighbors_a   = get_neighbors_up_to_depth(graph1,   node_a, 7)
        m_neighbors_a = get_neighbors_up_to_depth(graph2, m_node_a, 7)  
        loc_verified, date_verified, loc_neighbors_a, loc_m_neighbors_a, date_neighbors_a, date_m_neighbors_a= graph_neighbor_match(graph1, neighbors_a, graph2, m_neighbors_a)
        print ('<-------------------------------------------loc 1') 
        print (loc_neighbors_a)
        print ('--------------------------------------------loc 2') 
        print (loc_m_neighbors_a)
        print ('--------------------------------------------date 1') 
        print (date_neighbors_a)
        print ('--------------------------------------------date 2') 
        print (date_m_neighbors_a)
        print ('-------------------------------------------------->') 
        #pdb.set_trace()
        print ('---------- support or conflict ----------')
        if loc_verified and date_verified: 
            print ('support found')
            supports.append(node_a)
        else:
            print ('conflict detected')
            #pdb.set_trace()
            conflicts.append(node_a)
            conflict_dict[node_a]={'match':m_node_a,'loc_ver':loc_verified,'loc':loc_neighbors_a,'loc_m':loc_m_neighbors_a,'date_ver':date_verified,'date':date_neighbors_a,'date_m':date_m_neighbors_a}              
    return conflicts, supports ,conflict_dict

####


def verify_edges(all_nodes,graph1, graph2, matched_nodes, node_similarity_threshold,edge_similarity_threshold, vgraph,vtsrc):
    ve = 0
    mappings={}

    conflicts,supports,conflict_dict = check_all_neighbor_conflicts(graph1, graph2, matched_nodes, all_nodes)

    for edge1 in graph1.edges():
        node_a, node_b = edge1
 
        act1 = graph1.edges[edge1]['action']
        print('trying matching: << g1: '+ str(node_a)   + '-<' + str(act1) + '>-' + str(node_b) )
        
        m_node_a = find_closest_node(node_a, all_nodes, matched_nodes)
        m_node_b = find_closest_node(node_b, all_nodes, matched_nodes)
        
        
        
        
        if not m_node_a or not m_node_b:
            continue
        
        
        act2 = find_path_with_actions(graph2,m_node_a,m_node_b)
        if act2==1:
            pdb.set_trace()
        
        print('matching: << g1: '+ str(node_a)   + '-<' + str(act1) + '>-' + str(node_b) )
        print('             g2: '+ str(m_node_a) + '-<' + str(act2)  + '>-' + str(m_node_b)+'>>' )


        cur_edge_sim=calculate_node_similarity(encode_text(act1, tokenizer, bertmodel),encode_text(act2, tokenizer, bertmodel)).item()
        if cur_edge_sim>edge_similarity_threshold:    
            vgraph.edges[edge1]['action'] = graph1.edges[edge1]['action'] + '  <verified> by '  + vtsrc
            
            mappings[ve]={'src1':node_a,'src2':node_b,'tar1':m_node_a,'tar2':m_node_b}
            
            print('<<<-----------------verified edge----------------->>> ')
            ve += 1
        else:    
            print(str(m_node_a)+" , "+str(m_node_b)+"   <NOT connected in graph 2> as edge sim:" +str(cur_edge_sim))


    ##
    print('conflicts and supports')
    print(conflicts)
    print(supports) 
    print(mappings)
    #if len(conflicts)>0:
    #   print('-------------------------<conflict reduced evidence>')
    #   ve=0
    
    return conflicts,supports,ve, vgraph,mappings,conflict_dict


def is_empty_graph(graph):
    return len(graph.nodes) == 0 or len(graph.edges) == 0



def preprocess_node_name(node_name,ent_type_graph,context):
    det_e,det_d,det_s= search_gkb_topk(node_name,context,3)
    
    #det_e,det_d,det_s= search_gkb(node_name,context)
    print('<<--------')
    print(node_name)
    print(ent_type_graph)
    
    print('<---gkb ent')
    print(det_e)
    print(det_d)
    print(det_s)
    print(context)
    print('------>>')
    
    #
    if det_s>gkb_thres:
        node_text= node_name +' or ' + det_e +' is '+ ent_type_graph + ' . '+ det_d
        print('#########################################################################################gkb hit')
    else:
        node_text= node_name +' is '+ ent_type_graph
    
    print(node_text)
            
    return node_text.lower().strip()


def display_matchings(all_nodes, matched_nodes,node_label):
    for i, node in enumerate(all_nodes):
        try:
           matched_index = np.nonzero(matched_nodes[i, :]).item()
           mnode         = all_nodes[matched_index]
           try:
              gsrc=np.nonzero( node_label[all_nodes.index(node)]).item()
              mgsrc=np.nonzero( node_label[all_nodes.index(mnode)]).item()
           except:
              gsrc=1
              mgsrc=2
              print(str(node) +' node must be same as '+str(mnode))
           
           print( str(gsrc) +':' + str(node) +'->'+ str(mgsrc) +':' + str(mnode))

           print("\n")
        except:
           print('no match found for : < '+str(node)+' >')





def graph_similarity(graph1, context1, graph2, context2, node_similarity_threshold, edge_similarity_threshold,vtsrc):
    try:
        scr, vgraph,conflicts,supports,search_terms,vmaps,conflict_dict=graph_similarity_core(graph1, context1, graph2, context2, node_similarity_threshold, edge_similarity_threshold,vtsrc)
        return  scr, vgraph,conflicts,supports,search_terms,vmaps,conflict_dict
    except:
        return 0, nx.empty_graph(1, create_using=nx.DiGraph()) , [],[],' no st',{} ,{}
        
        
        
def assignment(node_similarities,node_sim_thres):
    node_similarities_np = node_similarities.numpy() if isinstance(node_similarities, torch.Tensor) else node_similarities
    row_indices = []
    col_indices = []
    for i in range(node_similarities_np.shape[0]):
      col_index = np.argmax(node_similarities_np[i, :])
      val=node_similarities_np[i,col_index]
      if val > node_sim_thres:
         col_indices.append(col_index)
         row_indices.append(i)
    return row_indices, col_indices 
                 
            
def graph_similarity_core(graph1, context1, graph2, context2, node_similarity_threshold, edge_similarity_threshold,vtsrc):
    
    
    ###--------------------------------------------------------------------quick exit if empty graph
    scr=0
    empt_g = nx.empty_graph(1, create_using=nx.DiGraph())
    if is_empty_graph(graph1) or is_empty_graph(graph2):
        print(' <<<<<<<<<< empty graph')
        return scr, empt_g ,[],[],''

    
    ####-----------------------------------------------------------------------nodematching
    all_nodes = list(set(graph1.nodes) | set(graph2.nodes))
    all_edges = list(set(graph1.edges) | set(graph2.edges))
    node_embeddings1 = torch.zeros((len(all_nodes), bertmodel.config.hidden_size))
    node_embeddings2 = torch.zeros((len(all_nodes), bertmodel.config.hidden_size))

    node_label        = torch.zeros((len(all_nodes), 3))


    for i, node in enumerate(all_nodes):
        if node in graph1.nodes:
            ent_type_graph1     = graph1.nodes[node].get('ent_type', 'THING')
            data_graph1            = graph1.nodes[node].get('data', 'THING')
            nodetext                   =ent_type_graph1 +' ' +data_graph1 
            
            #nodecontext1        = data_graph1  
            #nodetext            = preprocess_node_name(node,ent_type_graph1,nodecontext1)  ##not using sentence context
            
            node_embeddings1[i] = encode_text(nodetext, tokenizer, bertmodel)
            node_label[i,1]     = 1
        else:
            node_embeddings1[i] = torch.zeros(1, 768)

        if  node in graph2.nodes:
            ent_type_graph2     = graph2.nodes[node].get('ent_type', 'THING')
            data_graph2         = graph2.nodes[node].get('data', 'THING')
            nodetext            = node +' is '+ent_type_graph2 +' ' +data_graph2 
            
            #nodecontext2        = data_graph2 
            #nodetext            = preprocess_node_name(node,ent_type_graph2,nodecontext2)
            
            node_embeddings2[i] = encode_text(nodetext, tokenizer, bertmodel)
            node_label[i,2]     = 1
        else:
            node_embeddings2[i] = torch.zeros(1, 768)

    node_similarities = calculate_node_similarity(node_embeddings1, node_embeddings2)
    for i in range(node_similarities.shape[0]):
         if sum(node_label[i]).item() == 1 :                                             ## item in only one graph do this shit
             singmask               =node_label[ :,np.nonzero(node_label[i,:]).item()]  ## set one hot for neighbors of nodei, same graph
             node_similarities[i, :]=node_similarities[i, :]* (1-singmask)              ## similarity with nodes of the same graph set to 0
         else: #present in  both graphs both graphs, set as match 
             node_similarities[i, :]=node_similarities[i, :]* (0)  
             node_similarities[i, i] =1
 
    ###############hungarian
    
    #pdb.set_trace()
    node_similarities[node_similarities < node_similarity_threshold] = 0.01
    matched_nodes                                                    = torch.zeros_like(node_similarities)
    row_indices, col_indices                                         = linear_sum_assignment(1-node_similarities) #assignment(node_similarities,node_similarity_threshold)#
    
    
    #
    #

    for i in range(len(row_indices)):
      if node_label[row_indices[i],1].item() == 1: # this node is from claim
         #case1: if only present in one graph
         if len(np.nonzero(node_label[row_indices[i]]))==1 and len(np.nonzero(node_label[col_indices[i]]))==1:
            src=np.nonzero(node_label[row_indices[i]]).item()
            dst=np.nonzero(node_label[col_indices[i]]).item()
            if( src != dst ) and node_similarities[row_indices[i],col_indices[i]]>node_similarity_threshold:
               matched_nodes[row_indices[i],col_indices[i]]=1
               matched_nodes[col_indices[i],  row_indices[i]] = 1 
         #case2: if this node occurs in both graphs
         else:
            print('common node, thus connected')
            matched_nodes[row_indices[i],col_indices[i]]=1 
            matched_nodes[col_indices[i], row_indices[i]] = 1    

    ####-----------------------------------------------------------------------nodematching
    #pdb.set_trace()
    print('<-----------matches')
    print(context1)
    print_graph_info(graph1)
    print(context2)
    print_graph_info(graph2)
    display_matchings(all_nodes, matched_nodes,node_label)
    print('------------matches>')  
    #pdb.set_trace()
    
    vgraph                                       = graph1.copy()
    conflicts,supports,ve, vgraph,vmaps,conflict_dict = verify_edges(all_nodes,graph1, graph2, matched_nodes, node_similarity_threshold,edge_similarity_threshold, vgraph,vtsrc)



    #pdb.set_trace()
    ####search terms
    graph1C=copy.deepcopy(graph1)
    graph2C=copy.deepcopy(graph2)
    search_terms=get_search_terms_conditioned(graph1C, graph2C) ## expects graphs in er format 
    

    ### generate evidence
    if ve == 0:
        vgraph = empt_g
    else:
        for vedge in vgraph.edges():
            if '<verified>' in vgraph.edges[vedge]['action'].split(' '):
                print('<<<-----------------verified edge found-------------->>> ')
            else:
                vgraph.remove_edge(vedge[0], vedge[1])

        vgraph.remove_nodes_from(list(nx.isolates(vgraph)))
        scr=ve
    
    
    return scr, vgraph,conflicts,supports,search_terms,vmaps,conflict_dict

def zsal_vt(ransamidx,    img_p,       caps, evd_grid,evidence_v_p, evidence_vtxt, evidence_ttxt, vis_sim_thres, node_similarity_threshold,  edge_similarity_threshold, evidence_v_scr,evidence_v_l_scr,evidence_t_scr):
    
    print('building graphs-------------')
    #pdb.set_trace()
    claim_graph_img, CDG, cres     = get_graph_igc( caps)
    tcres=copy.deepcopy(cres)
    
    ###conditional or not
    ##pdb.set_trace()
    vevd_graph_img,  VDG, vres     = get_graph_igc_conditioned( evidence_vtxt ,caps, tcres)
    tevd_graph_img,  TDG, tres     = get_graph_igc_conditioned( evidence_ttxt ,caps, tcres)
    
    #pdb.set_trace()
    #vevd_graph_img,  VDG, vres     = get_graph_igc( evidence_vtxt)
    #tevd_graph_img,  TDG, tres     = get_graph_igc( evidence_ttxt)
	
	
    print('building graphs------------- <done>')
    #pdb.set_trace()
    
    try:
       print('vistxt graph')
       SV,EVG, con_V,sup_V,STV,vmaps,vconflict_dict=graph_similarity(CDG.copy(),caps,VDG.copy(),evidence_vtxt,node_similarity_threshold, edge_similarity_threshold,' <vis> ')
       print('text graph')
       ST,ETG, con_T,sup_T,STT,tmaps,tconflict_dict=graph_similarity(CDG.copy(),caps,TDG.copy(),evidence_ttxt,node_similarity_threshold, edge_similarity_threshold,' <txt> ')
    except:
       print('similariy failed !!!')
       pdb.set_trace()
    print('graph similarity checks------------- <done>')
    try:
        vis_sim,vis_sim_list=evidence_v_scr,evidence_v_l_scr
    except:
        print('image data issue: ')
        print(evidence_v_p)
        print(img_p) 
        vis_sim=0
        vis_sim_list=[]
    print('visual similarity checks------------- <done>')

 
 
    #######################################################################
    qafrac=0
    bin_pred=1
    exps  =' '
    cl_edg_count=max(1,CDG.number_of_edges())
    print('<') 
    print(ST)
    print('---')
    print(evidence_t_scr)
    print(cl_edg_count)
    if len(con_T)==0:
        ST=(ST/cl_edg_count)
    else:
        print('text conflict < ')
        print(con_T)
        ST=0
    if len(con_V)==0:
       SV=(SV/cl_edg_count)
    else:
       print('image conflict < ')
       print(con_V)
											  
       SV=0
     
    ###<tdx> do we need image conflict checks?
    print(SV)      
    print('---')
    print(vis_sim_list)
    print(vis_sim)
    print('>')
					
    topk=4
    vis_vpfocs=vis_sim_list[0:6]
    vscr_vit,vscr_place,vscr_face,vscr_obj,vscr_cap,vscr_sct=vis_vpfocs
    score_cat=['vit','place','faces','objects','caption','scene_text']
    vis_cuml, score_cat_indices = sum(sorted(vis_vpfocs)[-topk:]) / topk, [score_cat[i] for i in sorted(range(len(vis_vpfocs)), key=lambda i: vis_vpfocs[i], reverse=True)[:topk]]

    
    '''if dataset_name=='remiss':  
       abl_file='remiss_data/bcn19/ablation.npz'  
       
    if dataset_name=='ccnd':    
       abl_file='newsclip_data/ablation.npz'   

    all_res  =np.load(abl_file,allow_pickle=True)
    all_res  =all_res[all_res.files[0]]
    all_res  =all_res.flat[0]'''
    
    #all_res={}
    
    abl_sim_txt,abl_sim_vxv,  abl_gm_txt,abl_gm_txvt, abl_im_vxv, bin_pred=1,1,1,1,1,1
    xt=False
    xv=False
    texps='  '
    vexps='  '
    if evidence_ttxt!='no_text_evidence':
        xt=True
        texps='XT(V) Rejects'
        if evidence_t_scr  > txt_sim_thres:
           abl_sim_txt=0
           texps     +=  ' XT(V) sim verifies  '
        if (evidence_t_scr  > txt_sim_thres-0.04) and ( (ST > 0.3) or ( len(con_T)==0 and len(sup_T) > 1  ) ):
           abl_gm_txt     =0 
           texps     += '  XT(V) Graph verifies  '       
        if len(con_T)>0:
          texps += '+ XT(V) conflicts in location / date '
          abl_gm_txt     =1

    if  vscr_vit       > vis_sim_thres: 
        abl_sim_vxv=0  
        vexps     +=  ' XV(T) sim(vit) verifies  '  
    if  vis_cuml       > vis_sim_thres: 
        abl_im_vxv     =0 
        vexps     +=  ' XV(T) sim(vpfogs) verifies  '   
    if  ( (vis_cuml       > vis_sim_thres - 0.04) and (SV > 0.45)   ) or ( (vis_cuml > vis_sim_thres ) and ( SV > 0.1) )  : 
        abl_gm_txvt     =0     
        vexps     +=  ' XV(T) GM verifies  '   
    exps = texps +' - ' +vexps
    bin_pred=1
    if  (xt==True):
       bin_pred =(abl_gm_txt)
    bin_pred= bin_pred  and  (( abl_im_vxv ) and (abl_gm_txvt))
    
    print(exps)
    #pdb.set_trace()
    #all_res[ransamidx]={'xt':xt,'sim_txt':abl_sim_txt,'gmatch_txt':abl_gm_txt,'sim_vxv':abl_sim_vxv,'imatch_vxv':abl_im_vxv, 'gmatch_txvt':abl_gm_txvt, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
    
    #np.savez(abl_file,all_res=all_res)
    
    
    claim_graph_img_ANN=create_empty_image(200,200)     
								 
    EG=merge_edges_with_concatenation(ETG, EVG)
    cl_edg_count=CDG.number_of_edges()
    ev_edg_count=EG.number_of_edges()
    if ev_edg_count> 0 and  cl_edg_count >0: 
        qafrac      =ev_edg_count/cl_edg_count
    else:
        qafrac      =0
    try:
      claim_graph_img_ANN=draw_graph_networkx(EG)
    except:
      claim_graph_img_ANN=create_empty_image(200,200) 
    

    new_CDG, new_VDG ,new_TDG= color_mapped_nodes_wc(CDG, VDG, vmaps,TDG,tmaps,tconflict_dict)
    claim_graph_img_ANN=save_graph_image(new_CDG)
    vevd_graph_img_ANN =save_graph_image(new_VDG)
    tevd_graph_img_ANN =save_graph_image(new_TDG)
	
    #pdb.set_trace()
    
    save_outputs(bin_pred,ransamidx, cl_edg_count, qafrac, exps,img_p, claim_graph_img, claim_graph_img_ANN, evd_grid,evidence_v_p,vevd_graph_img,vevd_graph_img_ANN, tevd_graph_img,tevd_graph_img_ANN, vis_sim,vis_sim_list,evidence_vtxt,evidence_ttxt)
    
    print('zsal_success----------------<>')
    return STV,claim_graph_img_ANN, vevd_graph_img_ANN,SV,tevd_graph_img_ANN,ST		

