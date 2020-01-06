import tqdm
import networkx as nx
import argparse
import numpy as np
import multiprocessing
import graph_tool as gt
from graph_tool.centrality import betweenness

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--graph", help='bundled graph')
parser.add_argument("-l","--length",help="contig length")
parser.add_argument("-o","--output",help="output file")
args = parser.parse_args()
G = nx.Graph()
cpus = multiprocessing.cpu_count()
print('Using {} cpus'.format(cpus))

print('Loading bundled graph...')
with open(args.graph,'r') as f:
    for line in tqdm.tqdm(f, desc='Reading bundled'):
        attrs = line.split()
        G.add_edge(attrs[0],attrs[2],mean=float(attrs[4]),stdev=float(attrs[5]),bsize=int(attrs[6]),ori=attrs[1]+attrs[3])

node_set = set(G.nodes())

print('Loading contig lengths...')
contig_length = {}
with open(args.length,'r') as f:
    for line in tqdm.tqdm(f, desc='Reading lengths'):
        attrs = line.split()
        if attrs[0] in node_set:
            contig_length[attrs[0]] = int(attrs[1])

del node_set

nx.set_node_attributes(G,'length',contig_length)
repeat_nodes = {}


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes_iter(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges_iter(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes_iter(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges_iter(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG


def get_centrality(subg):
    #    centralities = nx.betweenness_centrality(subg)
    #    print(centralities)
    _g = nx2gt(subg)
    centralities, _ = betweenness(_g)
    v = centralities.get_array()
    mean = float(np.mean(v))
    stdev = float(np.std(v))
    for node in _g.vertices():
        if centralities[node] >= mean + 3*stdev:
            repeat_nodes[_g.vertex_properties['id'][node]] = centralities[node]


def centrality_wrapper(graph):
    n_comp = nx.number_connected_components(graph)
    print('The graph has {} components'.format(n_comp))
    for subg in tqdm.tqdm(nx.connected_component_subgraphs(graph), total=n_comp, desc='Component'):
        if len(subg.nodes()) >= 50:
            get_centrality(subg)


G_copy = G.copy()

print('Writing output...')
ofile = open(args.output,'w')
for i in xrange(3):
    centrality_wrapper(G_copy)
    for node in tqdm.tqdm(repeat_nodes, desc='Checking repeats'):
        if G_copy.has_node(node):
            G_copy.remove_node(node)
            ofile.write(str(node)+'\t'+str(repeat_nodes[node])+'\n')

#for u,v,data in G_copy.edges(data=True):
#    print u +"\t"+data[u][v]['ori'][0]+v+"\t"+data[u][v]['ori'][1]+"\t"+str(data[u][v]["mean"])+"\t"+str(data[u][v]["stdev"])+"\t"+str(data[u][v]["bsize"])
#nx.write_gml(G_copy,args.output)
