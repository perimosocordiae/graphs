import numpy as np
from graphs import Graph

__all__ = ['chunk_up', 'concat_trajectories']


def chunk_up(trajectories, chunk_size=None, directed=False):
  if chunk_size is None:
    chunk_lengths = list(map(len, trajectories))
  else:
    chunk_lengths = []
    for t in trajectories:
      chunk_lengths.extend(_chunk_traj_idxs(len(t), chunk_size))
  return concat_trajectories(chunk_lengths, directed=directed)


def concat_trajectories(traj_lengths, directed=False):
  P = []
  last_idx = 0
  for tl in traj_lengths:
    P.append(last_idx + _traj_pair_idxs(tl))
    last_idx += tl
  return Graph.from_edge_pairs(np.vstack(P), num_vertices=last_idx,
                               symmetric=(not directed))


def _traj_pair_idxs(traj_len):
  ii = np.arange(traj_len)
  pairs = np.transpose((ii[:-1], ii[1:]))
  return pairs


def _chunk_traj_idxs(traj_len, chunk_size):
  num_chunks, extra = divmod(traj_len, chunk_size)
  if num_chunks == 0:
    return [extra]
  c = [chunk_size] * num_chunks
  c[-1] += extra  # Add any leftovers to the last chunk.
  return c
