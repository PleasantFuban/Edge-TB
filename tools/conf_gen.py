import json
from collections import deque

import numpy as np


class HostEnv:
	def __init__ (self, _id, _switch, _bw, _type=0):
		self.switch = _switch
		self.bw = _bw
		self.id = _id
		self.type = _type
		self.layer_count = 0
		self.layer = []
		self.up_host = []
		self.down_count = []
		self.current_down = []
		self.down_host = []
		self.sync = []

		self.round = _round [_id - 1]
		self.local_epoch_num = _local_epoch_num [_id - 1]
		self.batch_size = _batch_size [_id - 1]
		self.learning_rate = _learning_rate [_id - 1]
		self.start_index = _start_index [_id - 1]
		self.end_index = _end_index [_id - 1]
		self.worker_fraction = _worker_fraction [_id - 1]

	def __hash__ (self):
		return hash (self.id)

	def to_json (self):
		_string = \
			'{\r\n' \
			+ '"switch": "' + format_path (self.switch, 10000) + '",\r\n' \
			+ '"bw": ' + str (self.bw) + ',\r\n' \
			+ '"type": ' + str (self.type) + ',\r\n' \
			+ '"layer_count": ' + str (self.layer_count) + ',\r\n' \
			+ '"layer": ' + str (self.layer [::-1]) + ',\r\n' \
			+ '"up_host": ' + str (self.up_host [::-1]) + ',\r\n' \
			+ '"down_count": ' + str (self.down_count [::-1]) + ',\r\n' \
			+ '"down_host": ' + str (self.down_host [::-1]) + ',\r\n' \
			+ '"sync": ' + str (self.sync [::-1]) + ',\r\n' \
			+ '"round": ' + str (self.round) + ',\r\n' \
			+ '"local_epoch_num": ' + str (self.local_epoch_num) + ',\r\n' \
			+ '"batch_size": ' + str (self.batch_size) + ',\r\n' \
			+ '"learning_rate": ' + str (self.learning_rate) + ',\r\n' \
			+ '"start_index": ' + str (self.start_index) + ',\r\n' \
			+ '"end_index": ' + str (self.end_index) + ',\r\n' \
			+ '"worker_fraction": ' + str (self.worker_fraction) + '\r\n' \
			+ '}\r\n'
		return _string


class SwitchEnv:
	def __init__ (self, _id):
		self.id = _id
		self.switch = {}
		self.node = {}
		self.n_hop = {}
		self.forward = {}

	def __hash__ (self):
		return hash (self.id)

	def to_json (self):
		_string = \
			'{\r\n' \
			+ '"switch": ' + str (self.switch) + ',\r\n' \
			+ '"node": ' + str (self.node) + ',\r\n' \
			+ '"forward": ' + str (self.forward) + '\r\n' \
			+ '}\r\n'
		return _string


def format_path (_id, start):
	host = 's-etree-'
	s_index = 0
	if start == 10000:
		while _id > _switch_number [s_index]:
			s_index = s_index + 1
	else:
		while _id > _host_number [s_index]:
			s_index = s_index + 1
	return 'http://' + host + str (s_index + 1) + ':' + str (start + _id)


def gen_host_env ():
	for host in _host_list:
		# 初始化一个env对象
		if host ['id'] not in host_env_map:
			host_env_map [host ['id']] = HostEnv (host ['id'], host_conn [host ['id'] - 1], host_bw [host ['id'] - 1])
		# 赋值
		env = host_env_map [host ['id']]
		env.layer_count = env.layer_count + 1
		env.layer.append (host ['layer'])
		# 连上父节点
		upper_id = upper_queue.popleft ()
		if upper_id == host ['id']:
			env.up_host.append ('self')
		elif upper_id == 'top':
			env.up_host.append ('top')
		else:
			env.up_host.append ('n' + str (upper_id))
		# 占位
		env.sync.append (host ['sync'])
		env.down_count.append (host ['dc'])
		env.current_down.append (0)
		# 是后面a[dc]个的父节点
		for _ in range (host ['dc']):
			upper_queue.append (host ['id'])
		# 让父节点连上自己
		if len (queue) != 0:
			u_e = host_env_map [queue.popleft ()]
			# 在父节点的第curr个子节点集合
			curr = 0
			while u_e.current_down [curr] == u_e.down_count [curr]:
				curr = curr + 1
			if curr == len (u_e.down_host):
				u_e.down_host.append ([])
			if u_e.id == host ['id']:
				u_e.down_host [curr].append ('self')
			else:
				u_e.down_host [curr].append ('n' + str (host ['id']))
			u_e.current_down [curr] = u_e.current_down [curr] + 1
		# 是后面a[dc]个的父节点
		for _ in range (host ['dc']):
			queue.append (host ['id'])
		# 无子节点占位
		if host ['dc'] == 0:
			host_env_map [host ['id']].down_host.append ([])

	for key in host_env_map:
		file_name = '../node/env/n' + str (key) + '.env'
		with open (file_name, 'w') as file:
			file.writelines (host_env_map [key].to_json ())


def gen_switch_env ():
	size, _ = switch_bw.shape
	for switch_id in range (size):
		env = switch_env_map [switch_id + 1] = SwitchEnv (switch_id + 1)
		for next_id in range (size):
			if switch_id == next_id:
				continue
			if switch_bw [switch_id] [next_id] != 0:
				env.switch [format_path (next_id + 1, 10000)] = switch_bw [switch_id] [next_id]
	for index in range (len (host_conn)):
		env = switch_env_map [host_conn [index]]
		env.node [format_path (index + 1, 8000)] = host_bw [index]
		env.forward ['n' + str (index + 1)] = format_path (index + 1, 8000)
		env.n_hop ['n' + str (index + 1)] = 1
	flag = True
	while flag:
		flag = False
		for key1 in switch_env_map:
			s1 = switch_env_map [key1]
			f1 = s1.forward
			for path2 in s1.switch:
				key2 = int (path2 [path2.rfind (':') + 1:]) - 10000
				if key1 == key2:
					continue
				s2 = switch_env_map [key2]
				f2 = s2.forward
				for host1 in f1:
					if host1 not in f2 or s2.n_hop [host1] > s1.n_hop [host1] + 1:
						flag = True
						f2 [host1] = format_path (key1, 10000)
						s2.n_hop [host1] = s1.n_hop [host1] + 1

	for key in switch_env_map:
		file_name = '../switch/env/s' + str (key) + '.env'
		with open (file_name, 'w') as file:
			file.writelines (switch_env_map [key].to_json ())


def gen_yml ():
	host_start_number = 1
	switch_start_number = 1
	for dep_index in range (len (_host_number)):
		# dep-的信息
		str_dep = '---\r\n' \
		          + 'apiVersion: apps/v1\r\n' \
		          + 'kind: Deployment\r\n' \
		          + 'metadata:\r\n' \
		          + '  name: d-etree-' + str (dep_index + 1) + '\r\n' \
		          + 'spec:\r\n' \
		          + '  selector:\r\n' \
		          + '    matchLabels:\r\n' \
		          + '      label: l-p-etree-' + str (dep_index + 1) + '\r\n' \
		          + '  template: \r\n' \
		          + '    metadata:\r\n' \
		          + '      labels:\r\n' \
		          + '        label: l-p-etree-' + str (dep_index + 1) + '\r\n' \
		          + '    spec:\r\n' \
		          + '      hostname: p-etree-' + str (dep_index + 1) + '\r\n' \
		          + '      containers:\r\n'
		# dep中每个host container的信息
		host_number = _host_number [dep_index]
		for host_index in range (host_start_number, host_number + 1):
			str_dep = str_dep \
			          + '      - name: n' + str (host_index) + '\r\n' \
			          + '        image: etree-node:v1.0\r\n' \
			          + '        imagePullPolicy: Never\r\n' \
			          + '        ports:\r\n' \
			          + '        - containerPort: ' + str (8000 + host_index) + '\r\n' \
			          + '        command: ["python3", "node/node.py"]\r\n' \
			          + '        env:\r\n' \
			          + '        - name: HOSTNAME\r\n' \
			          + '          value: "n' + str (host_index) + '"\r\n' \
			          + '        - name: PORT\r\n' \
			          + '          value: "' + str (8000 + host_index) + '"\r\n' \
			          + '        volumeMounts:\r\n' \
			          + '        - name: etree\r\n' \
			          + '          mountPath: /home/etree\r\n'
		# dep中每个switch container的信息
		switch_number = _switch_number [dep_index]
		for switch_index in range (switch_start_number, switch_number + 1):
			str_dep = str_dep \
			          + '      - name: s' + str (switch_index) + '\r\n' \
			          + '        image: etree-switch:v1.0\r\n' \
			          + '        imagePullPolicy: Never\r\n' \
			          + '        ports:\r\n' \
			          + '        - containerPort: ' + str (10000 + switch_index) + '\r\n' \
			          + '        command: ["python3", "switch/switch.py"]\r\n' \
			          + '        env:\r\n' \
			          + '        - name: HOSTNAME\r\n' \
			          + '          value: "s' + str (switch_index) + '"\r\n' \
			          + '        - name: PORT\r\n' \
			          + '          value: "' + str (10000 + switch_index) + '"\r\n' \
			          + '        volumeMounts:\r\n' \
			          + '        - name: etree\r\n' \
			          + '          mountPath: /home/etree\r\n'
		# dep中volume的信息
		str_dep = str_dep \
		          + '      volumes:\r\n' \
		          + '      - name: etree\r\n' \
		          + '        persistentVolumeClaim:\r\n' \
		          + '          claimName: pvc-etree\r\n'
		# svc的信息
		str_dep = str_dep \
		          + '---\r\n' \
		          + 'apiVersion: v1\r\n' \
		          + 'kind: Service\r\n' \
		          + 'metadata:\r\n' \
		          + '  name: s-etree-' + str (dep_index + 1) + '\r\n' \
		          + 'spec:\r\n' \
		          + '  selector:\r\n' \
		          + '    label: l-p-etree-' + str (dep_index + 1) + '\r\n' \
		          + '  type: NodePort\r\n' \
		          + '  ports:\r\n'
		# svc中每个host port的信息
		for host_index in range (host_start_number, host_number + 1):
			str_dep = str_dep \
			          + '  - name: n' + str (host_index) + '\r\n' \
			          + '    port: ' + str (8000 + host_index) + '\r\n' \
			          + '    targetPort: ' + str (8000 + host_index) + '\r\n'
		# svc中每个switch port的信息
		for switch_index in range (switch_start_number, switch_number + 1):
			str_dep = str_dep \
			          + '  - name: s' + str (switch_index) + '\r\n' \
			          + '    port: ' + str (10000 + switch_index) + '\r\n' \
			          + '    targetPort: ' + str (10000 + switch_index) + '\r\n' \
			          + '    nodePort: ' + str (32000 + switch_index) + '\r\n'
		# 写入一个yml文件
		with open ('../k8s/dep-' + str (dep_index + 1) + '.yml', 'w')as f:
			f.writelines (str_dep)
			f.close ()
		# 下一个文件用
		host_start_number = host_number + 1
		switch_start_number = switch_number + 1

	with open ('../k8s/dep.sh', 'w') as f:
		str_dep_sh = '#!/bin/bash\r\n'
		for i in range (len (_host_number)):
			str_dep_sh = str_dep_sh + 'kubectl delete -f dep-' + str (i + 1) + '.yml\r\n'
		for i in range (len (_host_number)):
			str_dep_sh = str_dep_sh + 'kubectl create -f dep-' + str (i + 1) + '.yml\r\n'
		f.writelines (str_dep_sh)
		f.close ()


def read_json (filename):
	_file = open (filename)
	_str = _file.read ().replace ('\n', '').replace ('\r', '').replace ('\'', '\"')
	_json = json.loads (_str)
	_file.close ()
	return _json


env_json = read_json ('env.txt')
_host = 's-etree-'
_switch_number = env_json ['switch_number']
_host_number = env_json ['host_number']

if len (_switch_number) != len (_host_number):
	print ("Error: switch_number.length != host_number.length")
	exit (-1)

_host_list = env_json ['host_list']
_round = env_json ['round']
_local_epoch_num = env_json ['local_epoch_num']
_batch_size = env_json ['batch_size']
_learning_rate = env_json ['learning_rate']
_start_index = env_json ['start_index']
_end_index = env_json ['end_index']
_worker_fraction = env_json ['worker_fraction']

host_env_map = {}
upper_queue = deque (['top'])
queue = deque ([])

conn_json = read_json ('host_conn.txt')
host_conn = conn_json ['conn']
host_bw = conn_json ['bw']

switch_env_map = {}
switch_bw = np.loadtxt ('switch_bw.txt')

gen_host_env ()
gen_switch_env ()
gen_yml ()
