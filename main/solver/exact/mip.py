# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

import pprint

from main.core import Solution
from main.core.environment import SolutionStepEnvironment
from main.solver.base_solver import Solver, SolverRegistry


@SolverRegistry.register(solver_name='mip', solver_type='exact')
class MipSolver(Solver):
    """
    An exact solver based on Mixed Integer Programming (MIP) with OR-Tools.

    References:
        - Mosharaf Chowdhury et al. "ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping". In TON, 2012.
    """
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super(MipSolver, self).__init__(controller, recorder, counter, logger, config, **kwargs)
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'mcf')
        self.k_shortest = kwargs.get('k_shortest', 10)
        self.META_BW = 9999
        self.MAX_TIME_IN_SECONDS = float(config.solver.get('mip_time_limit_seconds', 10))
        self.OR_SOLVER = 'mip'
        if self.OR_SOLVER == 'cp':
            self.solver_with_or_tools = self.solve_with_cp
        else:
            self.solver_with_or_tools = self.solve_with_mip


    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']
        self.solution = Solution.from_v_net(v_net)
        self.solver_with_or_tools(v_net, p_net)
        return self.solution


    def construct_resource_dict(self, v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict):

        def get_node_type(n_id):
            return 'p' if n_id < num_p_nodes else 'v'

        def get_node_resource_dict(n_id, n_attr_name):
            n_type = get_node_type(n_id)
            if n_type == 'p':
                return p_net.nodes[n_id][n_attr_name]
            elif n_type == 'v':
                return v_net.nodes[n_id - num_p_nodes][n_attr_name]

        def get_edge_resource_dict(e_id, e_attr_name):
            u, v = e_id
            u_type = get_node_type(u)
            v_type = get_node_type(v)
            # p-p
            if u_type == 'p' and v_type == 'p':
                if (u, v) in p_net.links:
                    return p_net.links[(u, v)][e_attr_name]
                return 0
            elif u_type == 'v' and v_type == 'v':
                return 0
            else:
                if u > v:
                    candidates = candidates_dict[u-num_p_nodes]
                    return self.META_BW if v in candidates else 0
                else:
                    candidates = candidates_dict[v-num_p_nodes]
                    return self.META_BW if u in candidates else 0
        
        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        node_resource_dict = {}
        for n_id in range(num_p_nodes+num_v_nodes):
            node_resource_dict[n_id] = {}
            for n_attr_name in n_attr_name_list:
                node_resource_dict[n_id][n_attr_name] = get_node_resource_dict(n_id, n_attr_name)

        edge_resource_dict = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                e_id = (n_id_a, n_id_b)
                edge_resource_dict[e_id] = {}
                for e_attr_name in e_attr_name_list:
                    edge_resource_dict[e_id][e_attr_name] = get_edge_resource_dict(e_id, e_attr_name)

        return node_resource_dict, edge_resource_dict


    def solve_with_cp(self, v_net, p_net):

        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        p_node_list = list(range(num_p_nodes))
        v_node_list = list(range(num_v_nodes))
        m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
        a_node_list = list(range(num_p_nodes + num_v_nodes))


        n_attr_name_list = ['cpu']
        e_attr_name_list = ['bw']
        candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        # {m: [p for p in p_node_list] for m in m_node_list}
        node_resource_dict, edge_resource_dict = self.construct_resource_dict(v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict)
        assert len(e_attr_name_list) == 1

        model = cp_model.CpModel()

        # x varibles
        x = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                x[(n_id_a, n_id_b)] = model.NewBoolVar(f'x({n_id_a, n_id_b})')
                # x[(n_id_a, n_id_b)] = model.NewIntervalVar(f'x({n_id_a, n_id_b})')

        # f varibles
        f = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                for v_edge in v_net.links:
                    f[(n_id_a, n_id_b, v_edge)] = model.NewIntVar(lb=0, ub=self.META_BW, name=f'f({n_id_a, n_id_b, v_edge})')

        # Objective
        model.Minimize(
            sum(f[(u, v, i)] for i in list(v_net.links) for u in list(p_net.nodes) for v in list(p_net.nodes))
            + \
            sum(x[m, w] * node_resource_dict[m][n_attr_name] for m in m_node_list for w in p_node_list for n_attr_name in n_attr_name_list)
        )

        # Capacity constraint
        for n_attr_name in n_attr_name_list:
            for m in m_node_list:
                for w in p_node_list:
                    if node_resource_dict[m][n_attr_name] == 0:
                        break
                    model.Add(x[(m,w)] * node_resource_dict[m][n_attr_name] <= node_resource_dict[w][n_attr_name])

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    sum_flow = sum([f[(u, v, i)] + f[(v, u, i)] for i in v_net.links])
                    model.Add(sum_flow <= edge_resource_dict[(u,v)][e_attr_name] * x[(u, v)])

        for e_attr_name in e_attr_name_list:
            for i in v_net.links:
                for n_id in a_node_list:
                    if n_id == i[0] + num_p_nodes:
                        model.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == 1 * v_net.links[i][e_attr_name]
                        )
                    elif n_id == i[1] + num_p_nodes:
                        model.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == -1 * v_net.links[i][e_attr_name]
                        )
                    else:
                        model.Add(
                            sum(f[(n_id, w, i)] for w in a_node_list) - sum(f[(w, n_id, i)] for w in a_node_list) == 0
                        )

        # Meta constraint
        for m in m_node_list:
            model.Add(sum(x[(m,w)] for w in candidates_dict[m-num_p_nodes]) == 1)

        for w in p_node_list:
            model.Add(sum(x[(m,w)] for m in m_node_list) <= 1)

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    model.Add(x[(u,v)] <= edge_resource_dict[(u,v)][e_attr_name])

        for u in a_node_list:
            for v in a_node_list:
                model.Add(x[(u,v)] == x[(v,u)])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.MAX_TIME_IN_SECONDS
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
            node_slots = {}
            node_slots_info = {}
            link_paths = {}
            link_paths_info = {}
            for u in a_node_list:
                for v in a_node_list:
                    if solver.Value(x[(u, v)]):
                        if (u, v) not in p_net.links and u in p_net.nodes:
                            node_slots[v-num_p_nodes] = u
                            node_slots_info[(v-num_p_nodes, u)] = {n_attr_name: v_net.nodes[v-num_p_nodes][n_attr_name] for n_attr_name in n_attr_name_list}
                        
            for i in v_net.links:
                link_paths[i] = []
                for u in a_node_list:
                    for v in a_node_list:
                            if solver.Value(f[(u, v, i)]) and u in p_net.nodes and v in p_net.nodes:
                                link_paths[i].append((u, v))
                                link_paths_info[(i, (u, v))] = {e_attr_name_list[0]: solver.Value(f[(u, v, i)])}
                                # print(f[(u, v, i)], solver.Value(f[(u, v, i)]))
            # print(sum(solver.Value(x[(i, j)]) for i in a_node_list for j in a_node_list))
            pprint.pprint(node_slots_info)
            pprint.pprint(link_paths_info)

            self.solution['result'] = True
            self.solution['node_slots'] = node_slots
            self.solution['link_paths'] = link_paths
            self.solution['node_slots_info'] = node_slots_info
            self.solution['link_paths_info'] = link_paths_info


    def solve_with_mip(self, v_net, p_net):

        num_p_nodes = p_net.number_of_nodes()
        num_v_nodes = v_net.number_of_nodes()

        p_node_list = list(range(num_p_nodes))
        v_node_list = list(range(num_v_nodes))
        m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
        a_node_list = list(range(num_p_nodes + num_v_nodes))


        n_attr_name_list = ['cpu','ram']
        e_attr_name_list = ['bw']
        candidates_dict = self.controller.construct_candidates_dict(v_net, p_net)
        if self._fix_endpoint_vnfs():
            self._add_endpoint_candidates(v_net, candidates_dict)
        # {m: [p for p in p_node_list] for m in m_node_list}
        node_resource_dict, edge_resource_dict = self.construct_resource_dict(v_net, p_net, n_attr_name_list, e_attr_name_list, candidates_dict)
        assert len(e_attr_name_list) == 1

        solver = pywraplp.Solver.CreateSolver('SCIP')

        # x varibles
        x = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                x[(n_id_a, n_id_b)] = solver.IntVar(lb=0, ub=1, name=f'x({n_id_a, n_id_b})')

        # f varibles
        f = {}
        for n_id_a in range(num_p_nodes+num_v_nodes):
            for n_id_b in range(num_p_nodes+num_v_nodes):
                for v_edge in v_net.links:
                    f[(n_id_a, n_id_b, v_edge)] = solver.IntVar(lb=0, ub=self.META_BW, name=f'f({n_id_a, n_id_b, v_edge})')

        # Objective
        flow_resource_cost = sum(
            f[(u, v, i)]
            for i in list(v_net.links)
            for u in list(p_net.nodes)
            for v in list(p_net.nodes)
        )
        node_resource_cost = sum(
            x[m, w] * node_resource_dict[m][n_attr_name]
            for m in m_node_list
            for w in p_node_list
            for n_attr_name in n_attr_name_list
        )
        resource_cost = flow_resource_cost + node_resource_cost
        latency_cost = None
        if self._use_resource_latency_objective() or self._enforce_latency_constraint():
            latency_cost = self._build_mip_latency_cost(v_net, p_net, f, x, num_p_nodes, p_node_list)
        if self._use_resource_latency_objective():
            solver.Minimize(
                self._objective_weight('resource_weight', 1.0) * resource_cost
                + self._objective_weight('latency_weight', 1.0) * latency_cost
            )
        else:
            solver.Minimize(resource_cost)

        # Capacity constraint
        for n_attr_name in n_attr_name_list:
            for m in m_node_list:
                for w in p_node_list:
                    if node_resource_dict[m][n_attr_name] == 0:
                        break
                    solver.Add(x[(m,w)] * node_resource_dict[m][n_attr_name] <= node_resource_dict[w][n_attr_name])

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    sum_flow = sum([f[(u, v, i)] + f[(v, u, i)] for i in v_net.links])
                    solver.Add(sum_flow <= edge_resource_dict[(u,v)][e_attr_name] * x[(u, v)])

        for e_attr_name in e_attr_name_list:
            for i in v_net.links:
                for n_id in a_node_list:
                    if n_id == i[0] + num_p_nodes:
                        solver.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == 1 * v_net.links[i][e_attr_name]
                        )
                    elif n_id == i[1] + num_p_nodes:
                        solver.Add(
                            sum(f[n_id, w, i] for w in a_node_list) - \
                            sum(f[w, n_id, i] for w in a_node_list) == -1 * v_net.links[i][e_attr_name]
                        )
                    else:
                        solver.Add(
                            sum(f[(n_id, w, i)] for w in a_node_list) - sum(f[(w, n_id, i)] for w in a_node_list) == 0
                        )

        # Meta constraint
        for m in m_node_list:
            solver.Add(sum(x[(m,w)] for w in candidates_dict[m-num_p_nodes]) == 1)

        for w in p_node_list:
            solver.Add(sum(x[(m,w)] for m in m_node_list) <= 1)
        if self._fix_endpoint_vnfs():
            v_start = 0
            v_end = num_v_nodes - 1
            p_start = v_net.get_graph_attribute("src")
            p_end = v_net.get_graph_attribute("dst")
            solver.Add(x[(v_start + num_p_nodes, p_start)] == 1)
            solver.Add(x[(v_end + num_p_nodes, p_end)] == 1)

        if self._enforce_latency_constraint() and latency_cost is not None:
            max_latency = v_net.graph.get('max_latency', None)
            if max_latency is not None:
                solver.Add(latency_cost <= float(max_latency) + self._latency_constraint_margin())

        for e_attr_name in e_attr_name_list:
            for u in a_node_list:
                for v in a_node_list:
                    solver.Add(x[(u,v)] <= edge_resource_dict[(u,v)][e_attr_name])

        for u in a_node_list:
            for v in a_node_list:
                solver.Add(x[(u,v)] == x[(v,u)])

        solver.SetTimeLimit(int(self.MAX_TIME_IN_SECONDS * 1000))
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE :
            # print('Solution:')
            # print('Objective value =', solver.Objective().Value())
            node_slots = {}
            node_slots_info = {}
            link_paths = {}
            link_paths_info = {}
            for u in a_node_list:
                for v in a_node_list:
                    if x[(u, v)].solution_value():
                        if (u, v) not in p_net.links and u in p_net.nodes:
                            node_slots[v-num_p_nodes] = u
                            node_slots_info[(v-num_p_nodes, u)] = {n_attr_name: v_net.nodes[v-num_p_nodes][n_attr_name] for n_attr_name in n_attr_name_list}
                        
            for i in v_net.links:
                link_paths[i] = []
                for u in a_node_list:
                    for v in a_node_list:
                            if f[(u, v, i)].solution_value() and u in p_net.nodes and v in p_net.nodes:
                                link_paths[i].append((u, v))
                                link_paths_info[(i, (u, v))] = {e_attr_name_list[0]: f[(u, v, i)].solution_value()}

            if self._debug_print_solution():
                pprint.pprint(node_slots_info)
                pprint.pprint(link_paths_info)

            self.solution['result'] = True
            self.solution['node_slots'] = node_slots
            self.solution['link_paths'] = link_paths
            self.solution['node_slots_info'] = node_slots_info
            self.solution['link_paths_info'] = link_paths_info

            total_latency = self.compute_total_latency(v_net, p_net, self.solution)
            self.solution['total_latency'] = total_latency
            max_latency = v_net.graph['max_latency']
            if max_latency is not None and total_latency > max_latency:
                self.solution.update({'route_result': False, 'result': False, 'description': 'exceed max_latency'})
                return False
        else:
            status_map = {
                pywraplp.Solver.OPTIMAL: 'OPTIMAL',
                pywraplp.Solver.FEASIBLE: 'FEASIBLE',
                pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
                pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
                pywraplp.Solver.ABNORMAL: 'ABNORMAL',
                pywraplp.Solver.MODEL_INVALID: 'MODEL_INVALID',
                pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
            }
            status_name = status_map.get(status, str(status))
            if status == pywraplp.Solver.NOT_SOLVED:
                description = 'MIP time limit no feasible solution'
            elif status == pywraplp.Solver.INFEASIBLE:
                description = 'MIP infeasible'
            else:
                description = f'MIP failed: {status_name}'
            self.solution.update({
                'result': False,
                'route_result': False,
                'description': description,
                'mip_status': status_name,
            })
            return False

    def _fix_endpoint_vnfs(self):
        return bool(self.config.solver.get('fix_endpoint_vnfs', False))

    def _include_endpoint_latency(self):
        return bool(self.config.solver.get('include_endpoint_latency', True))

    def _objective_config(self):
        cfg = self.config.solver.get('objective', {}) if hasattr(self.config.solver, 'get') else {}
        return cfg if cfg is not None else {}

    def _objective_name(self):
        return str(self._objective_config().get('name', 'resource')).lower()

    def _use_resource_latency_objective(self):
        return self._objective_name() in {'resource_latency', 'latency_resource', 'cost_latency'}

    def _enforce_latency_constraint(self):
        try:
            return bool(self._objective_config().get('enforce_latency_constraint', False))
        except Exception:
            return False

    def _latency_constraint_margin(self):
        try:
            return float(self._objective_config().get('latency_constraint_margin', 0.0))
        except Exception:
            return 0.0

    def _debug_print_solution(self):
        return bool(self.config.solver.get('mip_debug_print_solution', False))

    def _objective_weight(self, name, default):
        try:
            return float(self._objective_config().get(name, default))
        except Exception:
            return float(default)

    def _build_mip_latency_cost(self, v_net, p_net, f, x, num_p_nodes, p_node_list):
        """Linear latency term for MIP.

        For a virtual link i with bandwidth demand b_i, f[u,v,i] / b_i is the
        fraction of that demand carried by physical link (u,v).  Multiplying it
        by link latency gives a flow-weighted path-delay term.  Endpoint delay
        is also added when endpoint latency is enabled.
        """
        latency_cost = 0
        for v_link in v_net.links:
            bw_demand = float(v_net.links[v_link].get('bw', 1.0) or 1.0)
            bw_demand = max(bw_demand, 1e-9)
            for u in p_net.nodes:
                for v in p_net.nodes:
                    link_latency = float(self._link_latency(p_net, (u, v)) or 0.0)
                    if link_latency > 0:
                        latency_cost += f[(u, v, v_link)] * (link_latency / bw_demand)

        if self._include_endpoint_latency() and v_net.num_nodes > 0:
            src = v_net.get_graph_attribute("src")
            dst = v_net.get_graph_attribute("dst")
            first_v_node = 0
            last_v_node = v_net.num_nodes - 1
            for p_node in p_node_list:
                latency_cost += x[(first_v_node + num_p_nodes, p_node)] * self._shortest_latency(p_net, src, p_node)
                latency_cost += x[(last_v_node + num_p_nodes, p_node)] * self._shortest_latency(p_net, p_node, dst)
        return latency_cost

    def _shortest_latency(self, net, src, dst):
        if src is None or dst is None or src == dst:
            return 0.0
        try:
            import networkx as nx
            path = nx.shortest_path(net, src, dst, weight='ltc')
            latency = 0.0
            for u, v in zip(path[:-1], path[1:]):
                latency += float(self._link_latency(net, (u, v)) or 0.0)
            return latency
        except Exception:
            return 1e6

    def _add_endpoint_candidates(self, v_net, candidates_dict):
        endpoints = {0: v_net.get_graph_attribute("src"), v_net.num_nodes - 1: v_net.get_graph_attribute("dst")}
        for v_node_id, p_node_id in endpoints.items():
            if p_node_id not in candidates_dict[v_node_id]:
                candidates_dict[v_node_id].append(p_node_id)

    def compute_total_latency(self, v_net, p_net, solution):
        total_latency = 0.0
        for v_link in v_net.links:
            paths = solution['link_paths'].get(v_link) or solution['link_paths'].get((v_link[1], v_link[0]), [])
            for p_link in paths:
                total_latency += self._link_latency(p_net, p_link)
        if self._include_endpoint_latency() and solution['node_slots']:
            total_latency += self.compute_delay(p_net, v_net.get_graph_attribute("src"), solution['node_slots'][0])
            total_latency += self.compute_delay(p_net, solution['node_slots'][v_net.num_nodes - 1], v_net.get_graph_attribute("dst"))
        return total_latency

    def _link_latency(self, net, link):
        if link in net.edges:
            return net.edges[link].get('ltc', 0)
        reversed_link = (link[1], link[0])
        if reversed_link in net.edges:
            return net.edges[reversed_link].get('ltc', 0)
        return 0

    def compute_delay(self, net, src, dst):
        import networkx as nx
        from itertools import islice
        from main.utils import flatten_recurrent_dict, path_to_links
        shortest_paths = list(islice(nx.shortest_simple_paths(net, src, dst, weight='ltc'), 1))
        shortest_paths = shortest_paths[0]
        if len(shortest_paths) == 1:
            return 0
        p_link = path_to_links(shortest_paths)
        latency = 0.0
        # 閬嶅巻姣忔潯鐗╃悊璺緞
        for path in p_link:
            # 璁＄畻璇ョ墿鐞嗚矾寰勭殑鏃跺欢
            path_latency = 0.0
            # 鐩存帴鑾峰彇鏃跺欢
            if path in net.edges:  # 濡傛灉閾捐矾瀛樺湪
                # 浣跨敤 'ltc' 灞炴€ц幏鍙栨椂寤讹紝濡傛灉娌℃湁鍒欓粯璁や负0
                path_latency = net.edges[path].get('ltc', 0)  # 鑾峰彇 'ltc' 鏃跺欢灞炴€?            latency += path_latency
        return latency
