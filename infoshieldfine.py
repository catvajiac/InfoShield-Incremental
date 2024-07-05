import numpy as np
import copy
from collections import defaultdict
import time
import os

import poagraph
import seqgraphalignment
from utils import *

def all_sequences_cost(sequences, sid_arr, template):
    align_cost, cond_int = 0, []
    for sid in sid_arr:
        alignment = seqgraphalignment.SeqGraphAlignment(sequences[sid].data, template)
        ac, ci = alignment.alignment_encoding_cost()
        align_cost += ac
        cond_int.append(np.array(ci)[:, 0].astype(int))
    return align_cost + template.encoding_cost(), cond_int


def dichotomous_search(sequences, sid_arr, graph):
    m_l, m_r = 0, len(sid_arr) - 1
    cost_dict = {}
    while m_l < m_r:
        m_m = int((m_l + m_r) / 2)
        t1, t2 = graph.selectEdge(m_m - 1), graph.selectEdge(m_m + 1)
        if m_m - 1 not in cost_dict.keys():
            t1 = graph.selectEdge(m_m - 1)
            cost_dict[m_m - 1] = all_sequences_cost(sequences, sid_arr, t1)[0]
        asc1 = cost_dict[m_m - 1]
        if m_m + 1 not in cost_dict.keys():
            t2 = graph.selectEdge(m_m + 1)
            cost_dict[m_m + 1] = all_sequences_cost(sequences, sid_arr, t2)[0]
        asc2 = cost_dict[m_m + 1]
        if asc1 <= asc2:
            m_r = max(m_l, m_m - 1)
        else:
            m_l = min(m_r, m_m + 1)
    m_m = m_r if asc1 <= asc2 else m_l
    template = graph.selectEdge(m_m)
    return template


def slot_identify(sequences, sid_arr, template):
    _, cond_int = all_sequences_cost(sequences, sid_arr, template)
    result, e_arr, vh_arr = defaultdict(dict), [], []
    for idx, cond in enumerate(cond_int):
        startslot, count, tmp = True, 0, 0
        e_arr.append(len(cond[cond > 0]))
        vh_arr.append(len(cond))
        for c in cond:
            if startslot:
                if c == 1:
                    tmp, count = tmp + 1, count + 1
                    continue
                elif c == 3:
                    tmp += 1
                    continue
                if tmp != 0:
                    result[-1][idx] = tmp
                startslot, tmp = False, 0
                continue
            if c == 1:
                tmp, count = tmp + 1, count + 1
            elif c == 3:
                tmp += 1
            else:
                if tmp != 0:
                    result[count - 1][idx] = tmp
                count, tmp = count + 1, 0
        if tmp != 0:
            result[count][idx] = tmp

    slot_count, v = 0, template.nNodes
    for k, n in result.items():
        sp1 = log_star(slot_count) + slot_count * ceil(np.log2(v))
        sp2 = log_star(slot_count + 1) + (slot_count + 1) * ceil(np.log2(v))

        sc = len(cond_int) + np.sum([log_star(nn) + nn * word_cost() for nn in n.values()])
        uw1, uw2 = 0, 0
        for kk, vv in n.items():
            e, vh = e_arr[kk], vh_arr[kk]
            uw1 += e * ceil(np.log2(vh)) + 2 * e + e * word_cost()
            e -= vv
            uw2 += e * ceil(np.log2(vh)) + 2 * e + e * word_cost()

        if uw1 + sp1 > uw2 + sp2 + sc:
            try:
                if k == -1:
                    template.startslot = True
                else:
                    template.nodedict[k].slot = True
                slot_count += 1
                for kk, vv in n.items():
                    e_arr[kk] -= vv
            except:
                pass

    return template

class Template:
    def __init__(self, template, timetick, align_cost, conds, latest=True):
        self.template = template
        self.encoding_cost = template.encoding_cost()
        self.timetick = set(timetick)
        self.align_cost = align_cost
        self.total_cost = self.encoding_cost + self.align_cost
        self.conds = conds
        self.latest = latest

    def incoporate(self, seq, cost, cond):
        self.timetick.add(seq.timetick)
        self.align_cost += cost
        self.total_cost += cost
        self.conds[seq.sid] = cond

class InfoShield_MDL:
    def __init__(self):
        self.temp_arr = []
        self.sequences = {}
        self.noises = []

    def fit(self, lsh_label, sequences, timetick, incremental, old=False):
        self.sequences.update(sequences)
        sid_arr = np.concatenate([[s.sid for s in sequences.values()], self.noises])
        prev_total_cost = np.sum([self.sequences[sid].cost for sid in sid_arr]) + len(sid_arr)

        for temp in self.temp_arr:
            temp[1].latest = True
        self.noises = []

        while len(sid_arr) > 0:
            init_idx = 0
            init_seq = self.sequences[sid_arr[init_idx]]

            graph, gid = poagraph.POAGraph(init_seq.data, init_seq.sid), [init_idx]
            seq_total_cost = init_seq.cost
            graph_0 = copy.deepcopy(graph)

            ### Check whether there already exists suitable template
            if incremental:
                if len(self.temp_arr) > 0:
                    if not old:
                        order = np.array([0])
                        if len(self.temp_arr) > 1:
                            intersect = [len(list(set(init_seq.data) & set(temp.template.seq_by_arr()))) for temp in self.temp_arr]
                            order = np.argsort(intersect)[::-1]
                        skip = False
                        for idx, temp in zip(order, np.array(self.temp_arr)[order]):
                            alignment = seqgraphalignment.SeqGraphAlignment(init_seq.data, temp[1].template)
                            cost, cond = alignment.alignment_encoding_cost()
                            if cost < seq_total_cost:
                                self.temp_arr[idx][1].incoporate(init_seq, cost, cond)
                                self.temp_arr[idx][1].latest = False
                                sid_arr = np.delete(sid_arr, init_idx)
                                skip = True
                                break
                        if skip:
                            continue
                    else:
                        min_cost, min_cond, min_idx = seq_total_cost, None, -1
                        for idx, temp in enumerate(self.temp_arr):
                            alignment = seqgraphalignment.SeqGraphAlignment(init_seq.data, temp[1].template)
                            cost, cond = alignment.alignment_encoding_cost()
                            if cost < min_cost:
                                min_cost, min_cond, min_idx = cost, cond, idx
                        if min_idx != -1:
                            self.temp_arr[min_idx][1].incoporate(init_seq, min_cost, min_cond)
                            self.temp_arr[min_idx][1].latest = False
                            sid_arr = np.delete(sid_arr, init_idx)
                            continue

            ### If no existing template, go through general process
            for idx, label in enumerate(np.delete(sid_arr, init_idx)):
                this_seq = self.sequences[label]
                alignment = seqgraphalignment.SeqGraphAlignment(this_seq.data, graph_0)
                cost, _ = alignment.alignment_encoding_cost()
                seq_cost = this_seq.cost

                if cost < seq_cost:
                    gid.append(idx + 1)
                    alignment = seqgraphalignment.SeqGraphAlignment(this_seq.data, graph)
                    graph.incorporateSeqAlignment(alignment, this_seq.data, label)
                    seq_total_cost += seq_cost

            if len(gid) > 1:
                template = dichotomous_search(self.sequences, sid_arr[gid], graph)
                template = slot_identify(self.sequences, sid_arr[gid], template)

                align_cost, conds = 0, {}
                for id in gid:
                    sequence = self.sequences[sid_arr[id]].data
                    alignment = seqgraphalignment.SeqGraphAlignment(sequence, template)
                    cost, cond = alignment.alignment_encoding_cost()
                    align_cost += cost
                    conds[sid_arr[id]] = cond

                total_cost = prev_total_cost - seq_total_cost
                if len(self.temp_arr) != 0:
                    total_cost -= log_star(len(self.temp_arr)) + \
                                  len(sid_arr) * ceil(np.log2(len(self.temp_arr)))
                total_cost += (len(sid_arr) + len(gid)) * ceil(np.log2(len(self.temp_arr) + 1))
                total_cost += log_star(len(self.temp_arr) + 1) + \
                              template.encoding_cost() + \
                              align_cost

                ### Check whether total cost decreases by this template
                if total_cost < prev_total_cost * 1.2:
                    prev_total_cost = total_cost
                    self.temp_arr.append(({'batch_num': timetick, 'lsh_label': lsh_label}, Template(template, [timetick], align_cost, conds)))
                else:
                    self.noises.extend(sid_arr[gid])
            else:
                self.noises.extend(sid_arr[gid])
            ### Delete the assigned sequences
            sid_arr = np.delete(sid_arr, gid)
        return self

    def predict(self):
        init_cost = sum([v.cost for v in self.sequences.values()])
        final_cost, labels = len(self.sequences) + log_star(len(self.temp_arr)), list(self.sequences.keys())
        for temp in self.temp_arr:
            final_cost += temp.total_cost
            labels = [k for k in labels if k not in temp.conds.keys()]
        for label in labels:
            final_cost += self.sequences[label].cost
        # return (init_cost - final_cost) / init_cost
        return init_cost, final_cost

    def predict_all(self):
        init_cost, final_cost, s_num = [], [], []
        for temp in self.temp_arr:
            final_cost.append(int(temp.total_cost))
            init_cost.append(np.sum([self.sequences[k].cost for k in temp.conds.keys()]))
            s_num.append(len(temp.conds.keys()))
        return init_cost, final_cost, s_num

    def update(self):
        for idx in range(len(self.temp_arr)):
            if not self.temp_arr[idx][1].latest:
                sid_arr_new = np.array(list(self.temp_arr[idx].conds.keys()))
                graph = poagraph.POAGraph(self.sequences[sid_arr_new[0]].data, \
                                          self.sequences[sid_arr_new[0]].sid)

                align_cost = 0
                for sid in sid_arr_new[1:]:
                    seq = self.sequences[sid]
                    alignment = seqgraphalignment.SeqGraphAlignment(seq.data, graph)
                    cost, _ = alignment.alignment_encoding_cost()
                    graph.incorporateSeqAlignment(alignment, seq.data, sid)
                    align_cost += cost
                template = dichotomous_search(self.sequences, sid_arr_new, graph)
                template = slot_identify(self.sequences, sid_arr_new, template)

                if self.temp_arr[idx].total_cost > align_cost + template.encoding_cost():
                    continue

                timetick, conds = set(), {}
                for sid in sid_arr_new:
                    alignment = seqgraphalignment.SeqGraphAlignment(self.sequences[sid].data, template)
                    cost, cond = alignment.alignment_encoding_cost()
                    timetick.add(self.sequences[sid].timetick)
                    align_cost += cost
                    conds[sid] = cond
                self.temp_arr[idx] = Template(template, timetick, align_cost, conds)

    def slot_content(self):
        sc = {}
        for tid, temp in enumerate(self.temp_arr):
            tid += 1
            sc[tid] = {}
            sw_count_all = {}
            for label in temp.conds.keys():
                sw_count_all[label], sc[tid][label] = defaultdict(set), {}
                alignment = seqgraphalignment.SeqGraphAlignment(self.sequences[label].data, temp.template)
                sw_count = alignment.alignment_condition()[2]
                for k, v in sw_count.items():
                    for vv in v:
                        sw_count_all[label][k].add(vv)

                sorted_idx = sorted(list(sw_count_all[label].keys()))
                for idx in sorted_idx:
                    sc[tid][label][idx] = sw_count_all[label][idx]
        return sc

    def output(self, path):
        output_results(self.temp_arr, path)


def run_infoshieldfine(filename):
    # read data temporally
    sequences = read_temporal_data(filename)

    # run MDL for each batch_num
    infoshield_mdl = InfoShield_MDL()
    for batch_num in sequences.keys():
        for lsh_label, values in sequences[batch_num].items():
            infoshield_mdl.fit(lsh_label, values, batch_num, incremental=True, old=False)
        infoshield_mdl.update()
 
    # write data to CSV
    data = []

    for template_id, (metadata, template) in enumerate(infoshield_mdl.temp_arr):
        for id in template.conds.keys():
            data.append({
                'lsh_label': metadata['lsh_label'],
                'template_id': template_id,
                'ad_id': int(id)
            })

    results_filename = f"{''.join(filename.split('.')[:-1])}-final_results.csv"
    pd.DataFrame(data).to_csv(results_filename, index=False)
    print(f'Results written to CSV: {results_filename}')


if __name__ == '__main__':
    run_infoshieldfine()
