import textwrap

import numpy as np


class ShowerDAG:
    from heparchy.data import ShowerData as __ShowerData
    from pyvis.network import __Network

    def __init__(self, shower: __ShowerData, masks: dict):
        df = shower.to_pandas(data=['pdg', 'final', 'edges'])
        for key, mask in masks.items():
            df[key] = mask
        self.__df = df
        self.shower = shower

    @classmethod
    def from_numpy(cls, edges: np.ndarray, pmu: np.ndarray, final: np.ndarray,
                   pdg: np.ndarray, masks: dict):
        shower = self.__ShowerData(edges, pmu, final, pdg)
        return cls(shower, masks)

    def __pcl_listing(self, names):
        names = list(names)
        names.sort()
        return br.join(wrapper.wrap(', '.join(names))) + br

    self.__heading = lambda text: '<b>' + text + ':</b>' + br

    def __vtx_title(self, pdgs_in, pdgs_out, length=40):
        wrapper = textwrap.TextWrapper(
            width=length,
            break_long_words=False,
            break_on_hyphens=False,
            )
        num_in = len(pdgs_in)
        num_out = len(pdgs_out)
        names_in = map(pdg_to_name, pdgs_in)
        names_out = map(pdg_to_name, pdgs_out)
        br = '<br />'
        return str(
            heading(f'particles in ({num_in})')
            + pcl_listing(names_in)
            + br
            + heading(f'particles out ({num_out})')
            + pcl_listing(names_out)
            )

    def to_pyvis(self):
        pdg_to_name = lambda pdg: Particle.from_pdgid(pdg).name
        shower = Network('640px', '960px', notebook=True, directed=True)
        W_edges = partons['W'][['in_edge', 'out_edge']]
        for parton, parton_data in partons.items():
            colour = colours[parton]
            for num, pcl in parton_data.iterrows():
                node = int(pcl['in_edge'])
                next_node = int(pcl['out_edge'])
                in_edges = data.query('@node == out_edge')
                in_pdgs = in_edges['pdg']
                out_edges = data.query('@node == in_edge')
                out_pdgs = out_edges['pdg']
                title = vtx_title(pdgs_in=in_pdgs, pdgs_out=out_pdgs)
                label = ' '
                shape = 'dot'
                leaf_shape = 'star'
                if pcl['final']:
                    leaf_node = next_node
                    leaf_label = pdg_to_name(pcl['pdg'])
                    shower.add_node(leaf_node, label=leaf_label,
                                    shape=leaf_shape, color=colour)
                shower.add_node(node, label=label, title=title, shape=shape,
                                size=10, color=colour)

            in_edge = map(int, parton_data['in_edge'])
            out_edge = map(int, parton_data['out_edge'])
            edges = zip(in_edge, out_edge)

            shower.add_edges(edges)
        shower.show('shower.html')
