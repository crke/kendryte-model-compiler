import tensorflow as tf
import output_dir.network as n

def freeze_to_PB(session, out_path):
    LAST_DOT = out_path.find('/')
    out_dir = './' if LAST_DOT < 0 else out_path[:LAST_DOT+1]
    out_name = out_path[LAST_DOT+1:]

    inp = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')

    net = n.network_forward(inp)
    session.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()

    for node in graph_def.node:
        print(node.name)


    out_nodes_list = [n.name for n in session.graph_def.node]

    # print last node
    # print(out_nodes_list[-1])
    constant_graph = tf.graph_util.convert_variables_to_constants(session,
                                                               session.graph_def,
                                                               output_node_names=['18_conv/Relu'])

    # 18_conv/Relu
    # output_node_names=['18_conv/leaky_relu'])
    #tf.train.write_graph(constant_graph, out_dir, out_name, as_text=False)
    with open(out_path, 'wb') as f:
        f.write(constant_graph.SerializeToString())


n.load_data()
sess = tf.Session()
freeze_to_PB(sess, './pb_files/output.pb')
