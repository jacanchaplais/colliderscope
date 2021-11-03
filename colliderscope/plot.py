import textwrap

import numpy as np


class ShowerDAG:
    import pandas as __pd
    from pyvis.network import Network as __Network
    from mcpid.lookup import PdgRecords as __PdgRecords
    from colour import Color as __Color


    def __init__(self, edges: np.ndarray, pdg: np.ndarray, pt: np.ndarray,
                 final: np.ndarray, masks: dict = dict()):
        data = {'in_edge': edges['in'],
                'out_edge': edges['out'],
                'pt': pt,
                'final': final,
                'pdg': pdg,
                'background': False,
               }
        data.update(masks)
        self.__mask_keys = list(masks.keys())
        self.__df = self.__pd.DataFrame(data)
        self.__pdg_lookup = self.__PdgRecords()
        self.table = self.__df
        self.__br = '<br />'

    @classmethod
    def from_heparchy(cls, shower, masks: dict = dict()):
        return cls(edges=shower.edges, pt=shower.pt(), final=shower.final,
                   pdg=shower.pdg, masks=masks)

    def __pcl_listing(self, names, length=40):
        wrapper = textwrap.TextWrapper(
            width=length,
            break_long_words=False,
            break_on_hyphens=False,
            )
        names = list(names)
        names.sort()
        return self.__br.join(wrapper.wrap(', '.join(names))) + self.__br

    def __heading(self, text):
        return '<b>' + text + ':</b>' + self.__br

    def __vtx_title(self, names, out=False):
        num = len(names)
        way = 'out' if out == True else 'in'
        return str(
            self.__heading(f'particles {way} ({num})')
            + self.__pcl_listing(names)
            )

    def vis_nodes(self):
        df = self.__df.copy()
        df['name'] = self.__pdg_to_name(df['pdg'])
        edges_out = df.pivot_table(
                'name',
                index='in_edge',
                aggfunc=lambda x: self.__vtx_title(x, out=True),
                )
        edges_in = df.pivot_table(
                ['name', 'final'],
                index='out_edge',
                aggfunc={
                    'name': lambda x: self.__vtx_title(x, out=False),
                    'final': lambda x: x.size == 1 and x.all(),
                    }
                )
        final = edges_in['final']
        node_df = edges_out + edges_in
        node_df['final'] = final
        node_df.index.name = 'id'
        node_df.rename(columns={'name': 'title'}, inplace=True)
        node_df['size'] = 10
        node_df['shape'] = final.apply(
                lambda x: 'star' if x == True else 'dot')
        root_id = int(node_df.index.values[node_df['final'].isnull()])
        node_df.loc[(root_id, 'final')] = False
        node_df.loc[(root_id, 'title')] = 'start of event'
        node_df.loc[(root_id, 'shape')] = 'square'
        node_df.loc[(root_id, 'size')] = 20
        return node_df
        # node_df = edges_in.join(edges_out, lsuffix='_in', rsuffix='_out')
        # node_df.index.name = 'id'
        # node_df = node_df.dropna()
        # pcl_names = node_df.apply(title, axis=0)
        # title_series = title_series.apply(title, axis=1)
        # return title_series

    def __pdg_to_name(self, pdgs) -> list:
        if isinstance(pdgs, int):
            pdgs = (pdgs,)
        else:
            pdgs = tuple(pdgs)
        name = self.__pdg_lookup.properties(pdgs=pdgs, props=['name'])
        name = name['name']
        return name

    def to_pyvis(self, notebook=True):
        shower = self.__Network('640px', '960px', notebook=notebook, directed=True)
        partons = {mask_key: self.__df.query(mask_key)
                   for mask_key in self.__mask_keys
                   }
        colours = {mask_key: self.__Color(pick_for=mask_key).get_web()
                   for mask_key in self.__mask_keys}
        bg_q = ' and '.join(map(lambda key: f'(not {key})', self.__mask_keys))
        # print(bg_q)
        bg_df = self.__df.query(bg_q)
        bg_idxs = bg_df.index
        # print(bg_idxs)
        self.__df.loc[bg_idxs, 'background'] = True
        partons.update({'background': bg_df})
        colours.update({'background': '#000000'})
        shape = 'dot'
        leaf_shape = 'star'
        parton_edges = []
        for parton, parton_data in partons.items():
            colour = colours[parton]
            for num, pcl in parton_data.iterrows():
                node = int(pcl['in_edge'])
                next_node = int(pcl['out_edge'])
                in_edges = self.__df.query('@node == out_edge')
                in_pdgs = in_edges['pdg']
                out_edges = self.__df.query('@node == in_edge')
                out_pdgs = out_edges['pdg']
                title = self.__vtx_title(names=self.__pdg_to_name(in_pdgs), out=False)
                title += str(self.__br
                             + self.__vtx_title(names=self.__pdg_to_name(in_pdgs), out=False)
                             )
                label = ' '
                if pcl['final']:
                    leaf_node = next_node
                    leaf_label = self.__pdg_to_name(pcl['pdg'])[-1]
                    shower.add_node(leaf_node, label=leaf_label,
                                    shape=leaf_shape, color=colour)
                shower.add_node(node, label=label, title=title, shape=shape,
                                size=10, color=colour)
            in_edge = map(int, parton_data['in_edge'])
            out_edge = map(int, parton_data['out_edge'])
            parton_edges += [zip(in_edge, out_edge)]
        for edges in parton_edges:
            shower.add_edges(edges)
        # print(shower.nodes)
        # print(shower.edges)
        vis = shower.show('shower.html')
        self.shower = shower
        if notebook == True:
            return vis
