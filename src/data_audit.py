"""Read-only source audit; YOLO and original VisDrone DET text formats."""
from pathlib import Path
from collections import Counter, defaultdict
import math
import numpy as np
from PIL import Image
from .common import sha256, digest, write_json

CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def size_bin(w, h):
    if not np.isfinite([w, h]).all() or min(w, h) <= 0:
        return 'invalid'
    s = math.sqrt(w * h)
    for edge, name in ((2, 'underflow'), (8, 'VT'), (16, 'T'), (32, 'S'), (64, 'M')):
        if s < edge:
            return name
    return 'L'


def split_paths(root, split, fmt='yolo'):
    root = Path(root)
    if fmt == 'yolo':
        return root / 'images' / split, root / 'labels' / split
    folder = root / ('VisDrone2019-DET-' + ('test-dev' if split == 'test' else split))
    return folder / 'images', folder / 'annotations'


def load_records(root, split, fmt='yolo', image_dir=None):
    images, labels = split_paths(root, split, fmt)
    if image_dir:
        images, labels = Path(image_dir), None
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in EXTENSIONS) if images.exists() else []
    records = []
    for path in paths:
        rec = dict(image_id=path.stem, path=str(path.resolve()), objects=[], ignored=[], invalid=[],
                   gt_available=False, annotation_path=None)
        try:
            with Image.open(path) as im:
                rec['width'], rec['height'] = im.size
        except Exception as e:
            rec['error'] = str(e)
            records.append(rec)
            continue
        ann = labels / (path.stem + '.txt') if labels else None
        if ann and ann.exists():
            rec.update(gt_available=True, annotation_path=str(ann.resolve()))
            for line_no, line in enumerate(ann.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    v = [float(x) for x in line.replace(',', ' ').split()]
                    ignored = False
                    if fmt == 'yolo':
                        if len(v) != 5:
                            raise ValueError('expected class cx cy w h')
                        c, cx, cy, w, h = v
                        if c != int(c) or not 0 <= c < 10:
                            raise ValueError('class outside 0..9')
                        if not all(0 <= x <= 1 for x in (cx, cy, w, h)):
                            raise ValueError('normalized coordinate outside 0..1')
                        w, h = w * rec['width'], h * rec['height']
                        x, y = cx * rec['width'] - w / 2, cy * rec['height'] - h / 2
                        c = int(c) + 1
                    elif fmt == 'visdrone':
                        if len(v) < 8:
                            raise ValueError('expected eight VisDrone DET columns')
                        x, y, w, h, score, c = v[:6]
                        if c != int(c) or not 0 <= c <= 11:
                            raise ValueError('class outside 0..11')
                        c = int(c)
                        ignored = c in (0, 11) or score == 0
                    else:
                        raise ValueError('unsupported annotation format')
                    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                        raise ValueError('nonpositive or nonfinite box')
                    if x >= rec['width'] or y >= rec['height'] or x + w <= 0 or y + h <= 0:
                        raise ValueError('box outside image')
                    obj = dict(object_id=f'{path.stem}:{line_no}', class_id=c, bbox=[x,y,w,h],
                               size_bin=size_bin(w,h), source_line=line_no,
                               clipped_at_border=x < 0 or y < 0 or x+w > rec['width'] or y+h > rec['height'])
                    rec['ignored' if ignored else 'objects'].append(obj)
                except (ValueError, OverflowError) as e:
                    rec['invalid'].append(dict(line=line_no, reason=str(e), raw=line))
        records.append(rec)
    return records


def choose_split(root, requested='test', fmt='yolo'):
    records = load_records(root, requested, fmt)
    if records and all(r['gt_available'] for r in records):
        return requested, None
    if requested == 'test':
        val = load_records(root, 'val', fmt)
        if val and all(r['gt_available'] for r in val):
            return 'val', 'test GT absent or incomplete; fallback to val (no independent final test)'
    raise ValueError(f'{requested}: complete GT unavailable; use image viewer for unlabeled inputs')


def audit(root, fmt='yolo', output='reports/dataset_audit.json'):
    result = dict(root=str(root), format=fmt, splits={}, duplicates=[],
                  ignore_regions_preserved=fmt == 'visdrone',
                  classes={i+1: name for i, name in enumerate(CLASSES)})
    hashes = defaultdict(list)
    for split in ('train', 'val', 'test'):
        records = load_records(root, split, fmt)
        corrupt = []
        for r in records:
            h = sha256(r['path'])
            hashes[h].append(f'{split}/{r["image_id"]}')
            r['image_sha256'] = h
            r['annotation_sha256'] = sha256(r['annotation_path']) if r['annotation_path'] else None
            try:
                with Image.open(r['path']) as im:
                    im.load()
            except Exception as e:
                corrupt.append(dict(image_id=r['image_id'], error=str(e)))
        _, labels = split_paths(root, split, fmt)
        stems = {r['image_id'] for r in records}
        result['splits'][split] = dict(
            n_images=len(records), n_annotation_files=sum(r['gt_available'] for r in records),
            n_objects=sum(len(r['objects']) for r in records), n_ignored=sum(len(r['ignored']) for r in records),
            invalid=[dict(image_id=r['image_id'], **v) for r in records for v in r['invalid']],
            missing_annotations=[r['image_id'] for r in records if not r['gt_available']],
            orphan_annotations=[p.name for p in labels.glob('*.txt') if p.stem not in stems],
            corrupt_images=corrupt, duplicate_stems=len(records)-len(stems),
            bins=dict(Counter(o['size_bin'] for r in records for o in r['objects'])),
            class_counts=dict(Counter(o['class_id'] for r in records for o in r['objects'])),
            fingerprint=digest([(r['image_id'],r['image_sha256'],r['annotation_sha256']) for r in records]))
    result['duplicates'] = [v for v in hashes.values() if len(v) > 1]
    result['fingerprint'] = digest(result)
    write_json(output, result)
    return result


def convert_coco(root, split, fmt, output, exclude_invalid=False):
    records = load_records(root, split, fmt)
    if not records or any(not r['gt_available'] or r.get('error') or (r['invalid'] and not exclude_invalid) for r in records):
        raise ValueError('conversion requires complete, valid GT and readable images; inspect audit')
    write_json(str(output)+'.excluded.json', [dict(image_id=r['image_id'], **v) for r in records for v in r['invalid']])
    images, anns, mapping = [], [], []
    image_mapping=[]
    for image_id, r in enumerate(records, 1):
        image_mapping.append(dict(source_image_id=r['image_id'],source_path=r['path'],coco_image_id=image_id))
        images.append(dict(id=image_id, file_name=Path(r['path']).name, width=r['width'], height=r['height']))
        for o in r['objects'] + r['ignored']:
            # Class-independent ignored regions are retained in source sidecar; not fabricated as a class.
            if o in r['ignored']:
                mapping.append(dict(source_id=o['object_id'], coco_id=None, ignored=True, bbox=o['bbox']))
                continue
            aid = len(anns) + 1
            x,y,w,h = o['bbox']
            anns.append(dict(id=aid,image_id=image_id,category_id=o['class_id'],bbox=[x,y,w,h],area=w*h,iscrowd=0))
            mapping.append(dict(source_id=o['object_id'],source_annotation=r['annotation_path'],coco_id=aid))
    write_json(output, dict(info={'description':'CICS conversion; source IDs in sidecar'}, images=images,
                           annotations=anns,categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)]))
    write_json(str(output)+'.source_ids.json', mapping)
    write_json(str(output)+'.image_ids.json', image_mapping)
    return len(images), len(anns)
