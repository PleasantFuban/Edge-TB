from conf_generator import ConfGenerator, Conf


class RaConfGenerator(ConfGenerator):

    # noinspection PyUnresolvedReferences
    def gen_conf(self) -> None:
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            name = node['name']
            assert name not in self.node_conf_map, 'Duplicate node: ' + name
            self.node_conf_map[name] = Conf(name, sync=self.sync, epoch=node['epoch'], ring_size=len(self.nodes), pos=i)
