import pydot

def _get_label(node, orientation=None):
    meaning = {'left': '<', 'right': '>='}
    label = 'id: {}\nindex: {}'.format(
        str(id(node)),
        node['index']
    )
    if orientation:
        if isinstance(node[orientation], dict):
            label = '{} {}\nid: {}\nindex: {}'.format(
                meaning[orientation],
                node['value'],
                str(id(node[orientation])),
                node[orientation]['index']
            )
        else:
            label = '{} {}\nid: {}\nPREDICT: {}'.format(
                meaning[orientation],
                node['value'],
                str(id(node[orientation])),
                node[orientation]
            )
    return label

def _draw(graph, parent_label, node):
    if not isinstance(node, dict):
        return
    left_label = _get_label(node, 'left')
    right_label = _get_label(node, 'right')
    left_edge = pydot.Edge(parent_label, left_label)
    graph.add_edge(left_edge)
    right_edge = pydot.Edge(parent_label, right_label)
    graph.add_edge(right_edge)
    _draw(graph, left_label, node['left'])
    _draw(graph, right_label, node['right'])

def visualize(dTree, filename):
    root = dTree.root
    graph = pydot.Dot(graph_type='graph')
    root_label = _get_label(root)
    _draw(graph, root_label, root)
    graph.write_png(filename)
    print("done drawing tree to {}".format(filename))
