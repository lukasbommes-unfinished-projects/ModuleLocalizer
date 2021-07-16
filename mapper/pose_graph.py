def get_neighbors(pose_graph, node_id):
    """Returns the node_ids of direct neighbors of the node specified by node_id."""
    neighbors_keyframes = sorted(list(pose_graph.neighbors(node_id)))
    return neighbors_keyframes


# def get_secondary_neighbors(pose_graph, node_id):
#     """Returns secondary neighbors of node with node_id in pose graph."""
#     second_neighbors = []
#     for neighbor_list in [pose_graph.neighbors(n) for n in pose_graph.neighbors(node_id)]:
#         for n in neighbor_list:
#             if (n != node_id) and (n not in pose_graph.neighbors(node_id)):
#                 second_neighbors.append(n)
#     return list(set(second_neighbors))
