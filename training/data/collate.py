from typing import Any, Dict, List

from torch.utils.data._utils.collate import default_collate


def collate_with_cubify_instances(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return default_collate(batch)

    first = batch[0]
    if isinstance(first, dict) and "cubify_instances" in first:
        instances = [b.get("cubify_instances") for b in batch]
        batch_copy = []
        for b in batch:
            b_copy = b.copy()
            b_copy.pop("cubify_instances", None)
            batch_copy.append(b_copy)

        collated = default_collate(batch_copy)
        collated["cubify_instances"] = instances
        return collated

    return default_collate(batch)