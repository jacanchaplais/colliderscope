from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import graphicle as gcl


def pt_size(pt):
    log_pt = np.log(pt)
    size = (log_pt - np.min(log_pt)) / (np.max(log_pt) - np.min(log_pt))
    size = size + 0.1
    return size


def eta_phi(
        pmu: np.ndarray,
        pdg: np.ndarray,
        final: Optional[np.ndarray] = None,
        parent_masks: Optional[Dict[str, np.ndarray]] = None,
        eta_max: Optional[float] = 2.5,
        pt_min: Optional[float] = 0.5,
        ) -> None:
    pcls = gcl.ParticleSet.from_numpy(pmu=pmu, pdg=pdg, final=final)
    # mask for only observable particles
    vis_mask = gcl.MaskGroup()
    if final is not None:
        vis_mask['final'] = pcls.final
    if eta_max is not None:
        vis_mask['eta'] = np.abs(pcls.pmu.eta) < eta_max
    if pt_min is not None:
        vis_mask['pt'] = pcls.pmu.pt > pt_min
    # TODO: add edge case handling for when no masking applied
    vis_pcls = pcls[vis_mask.data]
    # format event into dataframe
    df = pd.DataFrame()
    df['pt'] = vis_pcls.pmu.pt
    df['eta'] = vis_pcls.pmu.eta
    df['phi'] = vis_pcls.pmu.phi
    df['name'] = vis_pcls.pdg.name
    df['size'] = pt_size(vis_pcls.pmu.pt)
    # label the particles according to their mask name
    df['parent'] = ''
    if parent_masks is not None:
        df['parent'] = 'background'
        for mask_name, parent_mask in parent_masks.items():
            vis_parent_mask = parent_mask[vis_mask.data]
            df.loc[(vis_parent_mask, 'parent')] = mask_name
    # create and display figure
    fig = px.scatter(df, x='eta', y='phi', color='parent', symbol='parent',
                     size='size', hover_data=['size', 'pt', 'name'])
    fig.show()
