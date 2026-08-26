# !pip install graphviz
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    def matrix_html(matrix):
        matrix = np.atleast_2d(matrix)
        rows = []
        for row in matrix:
            cells = ''.join(
                f'<TD BORDER="1" CELLPADDING="4">{value:.4g}</TD>'
                for value in row
            )
            rows.append(f'<TR>{cells}</TR>')
        return (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            + ''.join(rows) +
            '</TABLE>'
        )

    for n in nodes:
        label = (
            f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">'
            f'<TR><TD><B>{n.label}</B></TD></TR>'
            f'<TR><TD>{matrix_html(n.data)}</TD></TR>'
            f'<TR><TD>grad</TD></TR>'
            f'<TR><TD>{matrix_html(n.grad)}</TD></TR>'
            f'</TABLE>>'
        )

        dot.node(
            name=str(id(n)),
            label=label,
            shape='plain'
        )

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def draw_graph(end_targate):
    end_targate.backward()
    dot = draw_dot(end_targate)
    dot.body[:] = [
        line.replace("<B></B>", "<B>&#160;</B>")
        for line in dot.body
    ]
    dot.render()
    # display(dot)
