from conf_generator import ConfGenerator, Conf


class SspConfGenerator(ConfGenerator):

    # noinspection PyUnresolvedReferences
    def gen_conf(self) -> None:
        # assume that the first node is the master
        master = self.nodes[0]
        master_name = master['name']
        master_conf = self.node_conf_map[master_name] = Conf(master_name, duration=master['duration'], child_node=[])

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            name = node['name']
            assert name not in self.node_conf_map, 'Duplicate node: ' + name
            self.node_conf_map[name] = Conf(name, epoch=node['epoch'], staleness=self.staleness,
                                            master_node=master_name)
            master_conf.child_node.append(name)
