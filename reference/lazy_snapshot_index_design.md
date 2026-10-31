# Lazy Snapshot Index Architecture

## Problem
- 25 original HDF5 files × 4.3GB = **107GB** total
- 95% of that (72GB) is `raw_snapshot` + `snapshot_T1` pickle blobs (~20KB per transition)
- Training only needs numerical data = **57MB** (merged file)  
- BUT: targeted resets (JTT-style) need snapshot access at specific transitions

## Design: Two-Tier Storage with Lazy Index

```
┌─────────────────────────────────────────────────────────┐
│                  TRAINING (HOT PATH)                     │
│                                                          │
│  merged_all_v2.h5 (57MB, fully in RAM)                  │
│  ├── observations    [N, 17]   float32                  │
│  ├── actions         [N, 2]    float32                  │
│  ├── rewards         [N]       float32                  │
│  ├── z_t, z_t1       [N, 30]   float32                  │
│  ├── policy_id       [N]       int8                     │
│  ├── snap_file_id    [N]       uint8   ← NEW (1 byte)  │
│  └── snap_row_id     [N]       uint32  ← NEW (4 bytes) │
│                                                          │
│  Extra RAM: 5 bytes × 3.37M = 16MB                      │
└───────────────────────┬─────────────────────────────────┘
                        │ lazy lookup on reset
                        ▼
┌─────────────────────────────────────────────────────────┐
│               SNAPSHOT ARCHIVE (COLD PATH)               │
│                                                          │
│  snapshots/                                              │
│  ├── snap_00.h5   (snapshot-only, ~2.9GB → ~1.2GB)     │
│  ├── snap_01.h5                                         │
│  ├── ...                                                │
│  └── snap_24.h5                                         │
│                                                          │
│  Each file contains ONLY:                                │
│  ├── raw_snapshot   [M]   vlen_bytes  zstd-compressed   │
│  └── snapshot_T1    [M]   vlen_bytes  zstd-compressed   │
│                                                          │
│  Total on disk: ~30GB (vs 107GB), random-access via H5  │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. SnapshotStore (new class)

```python
class SnapshotStore:
    """Lazy-load snapshots from on-disk archive with LRU caching."""
    
    def __init__(self, archive_dir, file_manifest, cache_size=256):
        """
        Args:
            archive_dir:   Path to snapshots/ directory
            file_manifest: List of (filename, n_rows) — order = file_id
            cache_size:    Max snapshots to keep in LRU cache
        """
        self.archive_dir = archive_dir
        self.file_manifest = file_manifest
        self._file_handles = {}     # file_id → h5py.File (lazy-opened)
        self._cache = OrderedDict() # (file_id, row_id) → snapshot_bytes
        self._cache_size = cache_size
        
    def get(self, file_id: int, row_id: int) -> bytes:
        """Lazy-load a single snapshot (~20KB I/O, <1ms on SSD)."""
        key = (file_id, row_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
            
        # Open file handle (pooled)
        if file_id not in self._file_handles:
            fname = self.file_manifest[file_id][0]
            self._file_handles[file_id] = h5py.File(
                os.path.join(self.archive_dir, fname), 'r'
            )
            
        blob = self._file_handles[file_id]['raw_snapshot'][row_id]
        snap_bytes = bytes(blob)
        
        # LRU eviction
        self._cache[key] = snap_bytes
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
            
        return snap_bytes
    
    def get_by_buffer_idx(self, buffer, idx: int) -> dict:
        """Convenience: look up by merged buffer index."""
        file_id = buffer.snap_file_id[idx]
        row_id = buffer.snap_row_id[idx]
        snap_bytes = self.get(file_id, row_id)
        return pickle.loads(snap_bytes)
```

### 2. Enhanced merge_and_compress.py

The merge script adds `snap_file_id` and `snap_row_id`:

```python
# In merge loop:
out["snap_file_id"][offset:offset+n] = file_id      # uint8
out["snap_row_id"][offset:offset+n] = np.arange(n)   # uint32
```

### 3. Snapshot-Only Archive Creator

Strip numerical data from originals, keep only snapshot blobs:

```python
def create_snapshot_archive(src_file, dst_file):
    """Strip everything except raw_snapshot and snapshot_T1."""
    with h5py.File(src_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
        # Copy only blob datasets
        for key in ['raw_snapshot', 'snapshot_T1']:
            src.copy(key, dst)
```

### 4. Smart Snapshot Selection for JTT-Style Reset

Instead of uniform random reset, use loss-proportional sampling:

```python
def sample_targeted_reset(self, priorities=None):
    """JTT-inspired: reset to high-loss / high-uncertainty transitions.
    
    Args:
        priorities: np.ndarray of shape (fixed_dataset_size,)
                    Higher value = more likely to be selected.
                    Could be: TD error, policy entropy, Q-value disagreement.
    """
    if priorities is not None:
        valid = self._valid_snap_indices
        p = priorities[valid]
        p = p / p.sum()  # normalize to probability
        idx = np.random.choice(valid, p=p)
    else:
        idx = np.random.choice(self._valid_snap_indices)
    
    snap_dict = self.snapshot_store.get_by_buffer_idx(self, idx)
    return snap_dict, self.state[idx].copy(), self.z_t[idx].copy()
```

## Storage Comparison

| Approach | Training RAM | Disk Total | Random Access |
|----------|-------------|------------|---------------|
| Current (all in RAM) | 107GB | 107GB | ✅ instant |
| Current merged (no snap) | 57MB | 107GB | ❌ no snapshots |
| **Proposed (lazy index)** | **73MB** | **107GB** | **✅ ~1ms/read** |
| Proposed + archive strip | 73MB | ~35GB | ✅ ~1ms/read |

## Optional Optimization: Snapshot Subsampling

Not every transition needs a snapshot for JTT-style reset. 
Station-change boundaries are the natural reset points.

```
Episode: [A₁ A₂ A₃ B₁ B₂ C₁ C₂ C₃ C₄ D₁]
         Stop-A   Stop-B   Stop-C       Stop-D
         ↑snap    ↑snap    ↑snap        ↑snap
```

Store snapshots only at station-change events (~1/5 of transitions).
This would reduce snapshot archive: 107GB → ~21GB.

The `snap_row_id = UINT32_MAX` sentinel indicates "no snapshot available".
For JTT targeting, the priority-sampled index would snap to the nearest 
available snapshot.

## Implementation Priority

1. **Phase 1 (now)**: Add `snap_file_id` + `snap_row_id` to merged file. 
   Create `SnapshotStore` class. Keep original files as-is.
   → 0 extra disk, 16MB extra RAM, full snapshot access.

2. **Phase 2 (later)**: Create snapshot-only archive, delete numerical 
   data from originals → saves ~35GB disk.

3. **Phase 3 (if needed)**: Implement station-boundary subsampling in 
   collect_worker → saves ~85GB disk on next collection.
