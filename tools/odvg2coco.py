# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import json
import sys
from pathlib import Path


def to_coco_bbox(b):
    """
    Convert a bbox that is *likely* [x_min, y_min, x_max, y_max] to COCO [x, y, w, h].
    If that yields non-positive w/h, assume input was already [x, y, w, h] and return it.
    """
    if len(b) != 4:
        raise ValueError(f'bbox must have 4 numbers, got {b}')
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    if w <= 0:
        raise ValueError(f'bbox must have positive width, got {w}')
    if h <= 0:
        raise ValueError
    return [float(x1), float(y1), float(w), float(h)]


def convert(groups):
    """
    groups: list[list[image_dict]] as in the user's example.
    Returns (coco_dict, group_map)
    """
    images = []
    annotations = []
    categories = []
    category_name_to_id = {}
    group_map = {}  # group_idx -> [flat_image_indices]

    ann_id = 1
    img_id = 1

    # Build categories on the fly from category names
    def get_cat_id(name):
        nonlocal categories, category_name_to_id
        if name not in category_name_to_id:
            new_id = len(category_name_to_id) + 1
            category_name_to_id[name] = new_id
            categories.append({
                'id': new_id,
                'name': name,
                'supercategory': 'none'
            })
        return category_name_to_id[name]

    for g_idx, group in enumerate(groups):
        group_indices = []
        for item in group:
            # Create image entry
            img_entry = {
                'id': img_id,
                'file_name': item['filename'],
                'width': int(item['width']),
                'height': int(item['height']),
                # Non-standard but useful metadata can be kept:
                'tilt': float(item.get('tilt', 0.0)),
                'within_group_id': int(item.get('within_group_id', 0)),
                'group_index': g_idx,
            }
            images.append(img_entry)
            group_indices.append(img_id - 1)  # zero-based flat index

            # Add annotations for this image
            inst_list = item.get('detection', {}).get('instances', []) or []
            for inst in inst_list:
                bbox_coco = to_coco_bbox(inst['bbox'])
                cat_name = inst.get('category', str(inst.get('label', '0')))
                cat_id = get_cat_id(cat_name)
                w, h = bbox_coco[2], bbox_coco[3]
                area = float(max(0.0, w) * max(0.0, h))

                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': bbox_coco,
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': [],  # not provided
                })
                ann_id += 1

            img_id += 1

        group_map[g_idx] = group_indices

    coco = {
        'info': {
            'description': "Converted dataset",
            'version': "1.0",
            'year': 2026,
            'contributor': "",
            'date_created': ""
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    return coco, group_map


def main():
    if len(sys.argv) < 3:
        print('Usage: python odvg2coco.py <input.json> <output_prefix>')
        print(
            'Example: python odvg2coco.py input.json out  # writes out_coco.json and out_group.json'
        )
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_prefix = Path(sys.argv[2])

    in_path_dir = in_path.parent

    with in_path.open('r', encoding='utf-8') as f:
        groups = json.load(f)

    coco, group_map = convert(groups)

    coco_path = in_path_dir / out_prefix.with_name(out_prefix.name + '_coco.json')
    group_path = in_path_dir / out_prefix.with_name(out_prefix.name + '_group.json')

    with coco_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=4)

    with group_path.open('w', encoding='utf-8') as f:
        json.dump(group_map, f, ensure_ascii=False, indent=4)

    print(f'Wrote {coco_path}')
    print(f'Wrote {group_path}')


if __name__ == "__main__":
    main()
