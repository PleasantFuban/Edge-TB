import random
from flask import request
import ctl_utils
from controller_base import Controller
from manager_base import RuntimeManager


class RaRuntimeManager(RuntimeManager):
    current_round: int
    current_stage: str
    current_step: int
    ok_nodes: set
    finished_nodes: set

    def _hook_start(self):
        self.current_round = 1
        self.current_stage = 'train'
        self.current_step = 0
        self.ok_nodes = set()
        self.finished_nodes = set()

        for _, ip in self.physical_nodes.items():
            ctl_utils.send_data('GET', '/train', ip, self.dml_port)
        for _, ip_port in self.emulated_nodes.items():
            ctl_utils.send_data('GET', '/train', ip_port[0], ip_port[1])

    def action_finish(self) -> None:
        @self.app.route('/finish', methods=['POST'])
        def route_finish():
            name = request.args.get('name')
            self.finished_nodes.add(name)
            if len(self.finished_nodes) == self.node_count:
                self.send_log_request()
            return ''

    def load_actions(self) -> None:
        super(RaRuntimeManager, self).load_actions()

        @self.app.route('/ok', methods=['POST'])
        def route_ok():
            name = request.args.get('name')
            self.executor.submit(self.on_route_ok, name)
            return ''

    def on_route_ok(self, name):
        self.ok_nodes.add(name)
        if len(self.ok_nodes) == self.node_count:
            print('stage finished by all nodes: ' + self.current_stage + ' ' + str(self.current_step))
            self.ok_nodes.clear()

            if self.current_stage == 'train':
                self.current_stage = 'reduce'
                self.current_step = 1
            elif self.current_stage == 'reduce':
                if self.current_step == self.node_count - 1:
                    self.current_stage = 'gather'
                    self.current_step = 1
                else:
                    self.current_step += 1
            elif self.current_stage == 'gather':
                if self.current_step == self.node_count - 1:
                    self.current_stage = 'train'
                    self.current_step = 0
                else:
                    self.current_step += 1

            # send the command
            if self.current_stage == 'train':
                path = '/train'
            else:
                path = '/send?stage=' + self.current_stage + '&step=' + str(self.current_step)
            for _, ip in self.physical_nodes.items():
                ctl_utils.send_data('GET', path, ip, self.dml_port)
            for _, ip_port in self.emulated_nodes.items():
                ctl_utils.send_data('GET', path, ip_port[0], ip_port[1])


class RaController(Controller):

    # override
    def _init_link(self, links) -> None:
        node_names = list(self._nDict.keys())
        node_count = len(node_names)
        for i in range(len(node_names)):
            name = node_names[i]
            left = (i + node_count - 1) % node_count
            left_name = node_names[left]
            self.net.asymmetrical_link(self._nDict[name], self._nDict[left_name], bw=random.randint(200, 500),
                                       unit='mbps')


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='AppConfig yaml file', required=True)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    controller = RaController(dirname)
    controller.init(os.path.join(dirname, args.config))
    controller.set_runtime_manager(RaRuntimeManager(dirname))
    controller.run()
