from typing import Dict, Sequence, Optional, Any, List, Tuple
import os
import json
import shutil
import hashlib

import torch
import numpy as np
from numba import njit
import torch.distributed as dist

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from dataset.common import PuzzleDatasetMetadata

@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(self, data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), 
        aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        self._local_trajectories = {}
        self._trajectory_spool_dir: Optional[str] = None
        self._trajectory_spool_rank_dir: Optional[str] = None
        self._trajectory_spool_batch_index = 0

    def _trajectory_has_predictions(self, trajectory: Dict[str, Any]) -> bool:
        predictions = trajectory.get("predictions")
        if predictions is None:
            return False
        if isinstance(predictions, torch.Tensor):
            return predictions.shape[0] > 0
        return len(predictions) > 0
        
    def begin_eval(self):
        if not self.aggregated_voting:
            self._local_hmap = {}
            self._local_preds = {}
        # Trajectory metadata must be per-eval because curated trajectory files
        # are cleaned up after each evaluator result().
        self._local_trajectories = {}
        self._trajectory_spool_batch_index = 0
        if self._trajectory_spool_rank_dir is not None:
            shutil.rmtree(self._trajectory_spool_rank_dir, ignore_errors=True)
            os.makedirs(self._trajectory_spool_rank_dir, exist_ok=True)

    def set_trajectory_spool_dir(self, spool_dir: Optional[str], rank: int):
        self._trajectory_spool_dir = spool_dir
        self._trajectory_spool_rank_dir = None
        self._trajectory_spool_batch_index = 0
        if spool_dir is None:
            return
        self._trajectory_spool_rank_dir = os.path.join(spool_dir, f"rank_{rank}")
        shutil.rmtree(self._trajectory_spool_rank_dir, ignore_errors=True)
        os.makedirs(self._trajectory_spool_rank_dir, exist_ok=True)

    def _trajectory_key_info(self, trajectory: Dict[str, Any]):
        identifier = trajectory["puzzle_identifier"]
        if isinstance(identifier, torch.Tensor):
            identifier = int(identifier.item())

        name = self.identifier_map[identifier]
        orig_name, inverse_fn = inverse_aug(name)

        input_grid = inverse_fn(_crop(trajectory["inputs"].numpy()))
        input_hash = grid_hash(input_grid)

        final_pred = trajectory["predictions"][-1]
        final_pred_grid = inverse_fn(_crop(final_pred.numpy()))
        pred_hash = grid_hash(final_pred_grid)
        return orig_name, input_hash, pred_hash

    def update_trajectory_batch(self, trajectories: List[Dict[str, Any]]):
        if self._trajectory_spool_rank_dir is not None:
            self._spool_trajectory_batch(trajectories)
            return

        for trajectory in trajectories:
            if not self._trajectory_has_predictions(trajectory):
                continue

            orig_name, input_hash, pred_hash = self._trajectory_key_info(trajectory)
            final_q = 0.0
            if len(trajectory["q_halt_probs"]):
                q_value = trajectory["q_halt_probs"][-1]
                if isinstance(q_value, torch.Tensor):
                    final_q = float(q_value.item())
                else:
                    final_q = float(q_value)

            self._local_trajectories.setdefault(orig_name, {})
            self._local_trajectories[orig_name].setdefault(input_hash, {})
            existing_entry = self._local_trajectories[orig_name][input_hash].get(pred_hash)
            candidate_entry = {
                "pred_hash": pred_hash,
                "final_q": final_q,
                "trajectory": self._compact_trajectory_for_storage(trajectory),
            }
            if existing_entry is None or self._trajectory_entry_sort_key(candidate_entry) > self._trajectory_entry_sort_key(existing_entry):
                self._local_trajectories[orig_name][input_hash][pred_hash] = candidate_entry

    def _spool_trajectory_batch(self, trajectories: List[Dict[str, Any]]):
        if self._trajectory_spool_rank_dir is None:
            return

        for trajectory in trajectories:
            if not self._trajectory_has_predictions(trajectory):
                continue

            orig_name, input_hash, pred_hash = self._trajectory_key_info(trajectory)
            final_q = 0.0
            if len(trajectory["q_halt_probs"]):
                q_value = trajectory["q_halt_probs"][-1]
                if isinstance(q_value, torch.Tensor):
                    final_q = float(q_value.item())
                else:
                    final_q = float(q_value)

            compact = self._compact_trajectory_for_storage(trajectory)
            candidate_entry = {
                "pred_hash": pred_hash,
                "final_q": final_q,
                "is_final_correct": compact.get("is_final_correct", False),
                "is_any_step_correct": compact.get("is_any_step_correct", False),
                "first_correct_step": compact.get("first_correct_step"),
                "num_inference_steps": compact.get("num_inference_steps", 0),
            }

            self._local_trajectories.setdefault(orig_name, {})
            self._local_trajectories[orig_name].setdefault(input_hash, {})
            existing_entry = self._local_trajectories[orig_name][input_hash].get(pred_hash)
            if existing_entry is not None and self._trajectory_entry_sort_key(candidate_entry) <= self._trajectory_entry_sort_key(existing_entry):
                continue

            path = self._spooled_trajectory_path(orig_name, input_hash, pred_hash)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(compact, path)
            candidate_entry["path"] = path
            self._local_trajectories[orig_name][input_hash][pred_hash] = candidate_entry

    def _trajectory_entry_sort_key(self, entry: Dict[str, Any]):
        trajectory = entry.get("trajectory")
        if trajectory is None:
            return (
                entry.get("is_final_correct", False),
                entry.get("is_any_step_correct", False),
                entry.get("final_q", 0.0),
            )
        return (
            trajectory.get("is_final_correct", False),
            trajectory.get("is_any_step_correct", False),
            entry["final_q"],
        )

    def _compact_trajectory_for_storage(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(trajectory)
        for key in list(compact.keys()):
            if key.startswith("prediction_") and key[len("prediction_"):].isdigit():
                del compact[key]
                continue
            if key.startswith("q_") and key[len("q_"):].isdigit():
                del compact[key]
        return compact

    def _trajectory_entries_for_input(self, local_trajectories: Dict[str, Any], name: str, input_hash: int):
        stored = local_trajectories.get(name, {}).get(input_hash, {})
        if isinstance(stored, dict):
            return list(stored.values())
        return stored

    def _prediction_stats_for_input(self, local_preds: Dict[str, Any], name: str, input_hash: int):
        stored = local_preds.get(name, {}).get(input_hash, {})
        if isinstance(stored, dict):
            return stored.items()
        return stored

    def _spooled_trajectory_path(self, orig_name: str, input_hash: int, pred_hash: int) -> str:
        assert self._trajectory_spool_rank_dir is not None
        task_hash = hashlib.sha1(orig_name.encode("utf-8")).hexdigest()[:16]
        return os.path.join(
            self._trajectory_spool_rank_dir,
            f"task_{task_hash}",
            f"input_{input_hash}",
            f"pred_{pred_hash}.pt",
        )

    def _load_trajectory_entry(self, entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        if "trajectory" in entry:
            return entry["trajectory"]
        path = entry.get("path")
        if path is None:
            return None
        return torch.load(path, map_location="cpu")

    def _choose_representative_trajectory_entry(self, entries: List[Dict[str, Any]], pred_hash: int):
        candidates = [entry for entry in entries if entry["pred_hash"] == pred_hash]
        if not candidates:
            return None
        candidates.sort(key=self._trajectory_entry_sort_key, reverse=True)
        return candidates[0]

    def _choose_debug_trajectory_entry(self, entries: List[Dict[str, Any]]):
        candidates = [
            entry
            for entry in entries
            if (
                entry.get("trajectory", entry).get("is_any_step_correct", False)
                and not entry.get("trajectory", entry).get("is_final_correct", False)
            )
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda entry: (
                entry.get("trajectory", entry).get("first_correct_step") is None,
                entry.get("trajectory", entry).get("first_correct_step") or 10**9,
                -entry.get("trajectory", entry).get("num_inference_steps", 0),
            )
        )
        return candidates[0]

    def _save_curated_trajectories(
        self,
        save_path: str,
        filename: str,
        payload: Dict[str, Any],
    ) -> None:
        trajectories_dir = os.path.join(os.path.dirname(os.path.dirname(save_path)), "trajectories")
        os.makedirs(trajectories_dir, exist_ok=True)
        with open(os.path.join(trajectories_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(self._to_json_compatible(payload), handle)

    def _to_json_compatible(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._to_json_compatible(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_compatible(v) for v in value]
        return value
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        score_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        score_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()

        if score_values is None:
            if "q_continue_logits" in preds:
                score_values = (1.0 - preds["q_continue_logits"].to(torch.float64).sigmoid()).cpu()
            else:
                score_values = torch.ones(outputs["preds"].shape[0], dtype=torch.float64)

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}
        score_values = score_values[mask]

        # Get predictions
        for identifier, input, pred, q in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), score_values.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_hash = grid_hash(_inverse_fn(_crop(input)))
            
            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, {})
            stats = self._local_preds[orig_name][input_hash].setdefault(pred_hash, [0, 0.0])
            stats[0] += 1
            stats[1] += float(q)
    
    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
        trajectory_mode: str = "selected",
    ) -> Optional[Dict[str, float]]:
        # Gather predictions to rank 0 for voting
        if world_size > 1:
            global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(
                (self._local_hmap, self._local_preds, self._local_trajectories),
                global_hmap_preds,
                dst=0,
                group=group,
            )
        else:
            global_hmap_preds = [(self._local_hmap, self._local_preds, self._local_trajectories)]

        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        selected_trajectories = []
        debug_trajectories = []

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                
                p_map = {}
                trajectory_entries = []
                for hmap, preds, local_trajectories in global_hmap_preds:  # type: ignore
                    for stored_pred in self._prediction_stats_for_input(preds, name, input_hash):
                        if isinstance(stored_pred, tuple) and len(stored_pred) == 2 and isinstance(stored_pred[1], list):
                            h, stats = stored_pred
                            p_map.setdefault(h, [0, 0.0])
                            p_map[h][0] += stats[0]
                            p_map[h][1] += stats[1]
                        else:
                            h, q = stored_pred
                            p_map.setdefault(h, [0, 0.0])
                            p_map[h][0] += 1
                            p_map[h][1] += q
                    trajectory_entries.extend(self._trajectory_entries_for_input(local_trajectories, name, input_hash))
                        
                if not len(p_map):
                    print (f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                    
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
                selected_hash, selected_stats = p_map[0]
                selected_correct = selected_hash == label_hash

                if trajectory_mode in {"selected", "debug"}:
                    selected_entry = self._choose_representative_trajectory_entry(trajectory_entries, selected_hash)
                    selected_trajectory = self._load_trajectory_entry(selected_entry)
                    if selected_trajectory is not None:
                        payload = dict(selected_trajectory)
                        payload["task_name"] = name
                        payload["input_hash"] = input_hash
                        payload["selected_pred_hash"] = selected_hash
                        payload["selected_vote_count"] = selected_stats[0]
                        payload["selected_mean_score"] = selected_stats[1]
                        payload["selected_is_correct"] = selected_correct
                        selected_trajectories.append(payload)
                if trajectory_mode == "debug" and not selected_correct:
                    debug_entry = self._choose_debug_trajectory_entry(trajectory_entries)
                    debug_trajectory = self._load_trajectory_entry(debug_entry)
                    if debug_trajectory is not None:
                        payload = dict(debug_trajectory)
                        payload["task_name"] = name
                        payload["input_hash"] = input_hash
                        payload["selected_pred_hash"] = selected_hash
                        payload["selected_vote_count"] = selected_stats[0]
                        payload["selected_mean_score"] = selected_stats[1]
                        payload["selected_is_correct"] = selected_correct
                        payload["label_hash"] = label_hash
                        debug_trajectories.append(payload)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok
                    
                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds, local_trajectories in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                        
                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])
                
                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)
            if trajectory_mode in {"selected", "debug"} and len(selected_trajectories):
                self._save_curated_trajectories(
                    save_path,
                    "selected.json",
                    {
                        "save_mode": "selected",
                        "trajectories": selected_trajectories,
                    },
                )
            if trajectory_mode == "debug" and len(debug_trajectories):
                self._save_curated_trajectories(
                    save_path,
                    "debug.json",
                    {
                        "save_mode": "debug",
                        "trajectories": debug_trajectories,
                    },
                )
        if self._trajectory_spool_dir is not None:
            shutil.rmtree(self._trajectory_spool_dir, ignore_errors=True)

        # Final result
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}

        return all_results
