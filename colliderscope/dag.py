import textwrap

import numpy as np


class ShowerDAG:
    import pandas as __pd
    from pyvis.network import Network as __Network
    from mcpid.lookup import PdgRecords as __PdgRecords
    from colour import Color as __Color
    import networkx as __nx


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
        self.__eps = 1e-8

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

    def __hex_to_rgb_column(self, hex_col: np.ndarray) -> np.ndarray:
        colors, color_idxs = np.unique(hex_col, return_inverse=True)
        rgbs = np.array([self.__Color(str(hex_)).rgb for hex_ in colors])
        rgb_col = rgbs[color_idxs]
        return rgb_col

    def __rgb_to_hex_column(self, rgb_col: np.ndarray) -> np.ndarray:
        colors, color_idxs = np.unique(rgb_col, axis=0, return_inverse=True)
        hexs = np.array([self.__Color(rgb=tuple(rgb)).hex for rgb in colors])
        hex_col = hexs[color_idxs]
        return hex_col

    def __add_hex_columns(self, df_cols: list) -> np.ndarray:
        rgb_cols = [self.__hex_to_rgb_column(hex_col) for hex_col in df_cols]
        rgb_sum = np.sum(rgb_cols, axis=0)
        norm = np.linalg.norm(rgb_sum, axis=1, keepdims=True)
        rgb_mix = np.divide(rgb_sum, norm, where=(np.abs(norm)>self.__eps))
        hex_mix = self.__rgb_to_hex_column(rgb_mix)
        return hex_mix

    def __kamada_kawai_df(self):
        edges = self.__df[['in_edge', 'out_edge']]
        edges = edges.to_numpy()
        shower = self.__nx.convert.from_edgelist(edges)
        num_nodes = len(np.unique(edges))
        diameter_scale = 4.0 * float(num_nodes)
        pos = self.__nx.drawing.layout.kamada_kawai_layout(
                shower, scale=diameter_scale)
        pos_df = self.__pd.DataFrame(pos)
        pos_df = pos_df.T
        pos_df.index.name = 'id'
        pos_df = pos_df.rename(columns={0: 'x', 1: 'y'})
        return pos_df

    def _vis_nodes(self, kamada_kawai=False):
        # prevent accidentally mutation of instance data
        df = self.__df.copy()
        # add column of names
        df['name'] = self.__pdg_to_name(df['pdg'])
        # node colours
        mask_keys = self.__mask_keys
        red_color = self.__Color('red') 
        blue_color = self.__Color('blue') 
        color_range = red_color.range_to(blue_color, len(mask_keys))
        colors = {ma_key: color.get_web()
                  for ma_key, color in zip(mask_keys, color_range)}
        # construct vertex titles and whether vertices are leaves
        aggfuncs = {'name': lambda x: self.__vtx_title(x, out=True)}
        hrtg_agg_funcs = {ma_key: lambda x: x.any() for ma_key in mask_keys}
        aggfuncs.update(hrtg_agg_funcs)
        edges_out = df.pivot_table(
                ['name'] + list(mask_keys),
                index='in_edge',
                aggfunc=aggfuncs,
                )
        aggfuncs={
            'name': lambda x: self.__vtx_title(x, out=False),
            'final': lambda x: x.size == 1 and x.all(),
            }
        aggfuncs.update(hrtg_agg_funcs)
        edges_in = df.pivot_table(
                ['name', 'final'] + list(mask_keys),
                index='out_edge',
                aggfunc=aggfuncs,
                )
        # gather labels for leaf nodes
        labels = df.query('final')
        labels = labels.pivot_table(
                values='name', index='out_edge', aggfunc=lambda x: x)
        labels.index.name = 'id'
        labels.rename(columns={'name': 'label'}, inplace=True)
        final = edges_in['final']
        # append the titles for edges in and out of the vertices
        node_df = edges_in.join(edges_out, lsuffix='_in', rsuffix='_out',
                                how='outer')
        for ma_key in mask_keys:
            ma_key = str(ma_key)
            key_in = ma_key + '_in'
            key_out = ma_key + '_out'
            color_mask = node_df[key_in] | node_df[key_out]
            node_df[ma_key] = '#000000'
            node_df.loc[(color_mask, ma_key)] = colors[ma_key]
            node_df.drop(columns=[key_in, key_out], inplace=True)
        color_blend = self.__add_hex_columns(
                [node_df[mask_key] for mask_key in mask_keys])
        node_df.drop(columns=mask_keys, inplace=True)
        node_df['color'] = color_blend
        node_df['title'] = node_df['name_in'] + node_df['name_out']
        node_df.drop(columns=['name_in', 'name_out'], inplace=True)
        node_df['final'] = final
        # set the appropriate column headers and index names for pyvis
        node_df.index.name = 'id'
        node_df.rename(columns={'name': 'title'}, inplace=True)
        # add labels for final state particles
        node_df = node_df.join(labels, how='outer')
        # replace null values of titles and labels with empty strings
        node_df['title'].fillna(value='', inplace=True)
        node_df['label'].fillna(value='', inplace=True)
        # set visualisation defaults for nodes
        node_df['size'] = 10
        node_df['shape'] = final.apply(
                lambda x: 'star' if x == True else 'dot')
        # settings for root node (start of event)
        root_id = int(node_df.index.values[node_df['final'].isnull()])
        node_df.loc[(root_id, 'final')] = False
        node_df.loc[(root_id, 'title')] = 'start of event'
        node_df.loc[(root_id, 'shape')] = 'square'
        node_df.loc[(root_id, 'size')] = 20
        node_df.drop(columns='final', inplace=True)
        if kamada_kawai == True:
            pos_df = self.__kamada_kawai_df()
            node_df = node_df.join(pos_df)
        node_df.reset_index(inplace=True)
        return node_df.to_dict('records')

    def _vis_edges(self):
        df = self.__df[['in_edge', 'out_edge']].copy()
        df.rename(columns={'in_edge': 'from', 'out_edge': 'to'}, inplace=True)
        df['arrows'] = 'to'
        return df.to_dict('records')

    def __pdg_to_name(self, pdgs) -> list:
        if isinstance(pdgs, int):
            pdgs = (pdgs,)
        else:
            pdgs = tuple(pdgs)
        name = self.__pdg_lookup.properties(pdgs=pdgs, props=['name'])
        name = name['name']
        return name

    def to_pyvis(self, name: str, notebook: bool=True):
        fname = f'{name}.html'
        shower = self.__Network(
                '640px', '960px', notebook=notebook, directed=True)
        shower.nodes = self._vis_nodes()
        shower.edges = self._vis_edges()
        self.shower = shower
        if notebook == True:
            vis = shower.show(fname)
            return vis
        else:
            shower.save_graph(fname)
