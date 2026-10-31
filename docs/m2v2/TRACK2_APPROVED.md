# Track 2: approved by the user on 2026-10-29 ("Yes, download")

- **Source:** HF dataset `LuluCao/KungfuAthleteBot`, pinned at commit ff09f3795dc99bef73b21413b03fa8470917a22c.
- **License:** the card says `license: apache-2.0`; the paper says MIT. There is no non-commercial clause. CREDITS must cite
  the paper and say "MIT per paper / Apache-2.0 per HF card".
- **What was downloaded:** only the `org_smooth_mj/<id>.npz` files of fist-form clips, meaning titles containing 拳.
  - IDs 196-206 are excluded: those are the kfab duan-4 and duan-6 routines, which hold the train, val and test scenes plus
    the quarantined 206.
  - That leaves 80 clips from 22 source videos, about 451 MB. No mp4 files and no jump subset were downloaded.
- **Location:** `C:\Users\dundy\Downloads\projects\kfab_data\`. This is untrusted data in its own folder.
  - `npz/<id>.npz`: the clips.
  - `action_titles.csv`: the full index.
  - `fist_ids.txt`
  - `manifest_expected.tsv`: id, path, size and LFS sha256 for each clip.
  - `fetch.sh` and `fetch.log`
- **Hygiene: sealed processing per plan 6.2.**
  - Partition by **source video title** only, before any motion is looked at, with a committed seed: 60% train pool, 20%
    val2, 20% test2.
  - The val2 and test2 motion is touched only by the sealed eligibility script, which writes aggregate eligibility only.
  - Load every npz with `np.load(..., allow_pickle=False)`.
