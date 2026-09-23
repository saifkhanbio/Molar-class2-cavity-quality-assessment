#!/usr/bin/env python3
"""Surface-connected cavity discovery with optional registered image-mask support.

Uses the unchanged final_cav_con_comb_smooth measurement/export functions.
Occlusal depths retain max(raw depth - 0.75 mm, 0); proximal depths have no offset.
"""

import argparse
import csv
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import final_cav_con_comb_smooth as measurement

VERSION = "1.0"


def mesh_geometry(vectors):
    """Weld shared STL vertices and build edge-based triangle adjacency."""
    vectors = np.asarray(vectors, float)
    cross = np.cross(vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(vectors).all(axis=(1, 2)) & (double_area > 1e-10)
    vectors = vectors[valid]
    # Welding tolerance is 1e-5 mm; no smoothing is applied to coordinates.
    _, first, inverse = np.unique(np.round(vectors.reshape(-1, 3), 5), axis=0,
                                  return_index=True, return_inverse=True)
    vertices = vectors.reshape(-1, 3)[first]
    faces = inverse.reshape(-1, 3)
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    vectors, faces = vectors[np.sort(unique)], faces[np.sort(unique)]
    cross = np.cross(vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normals = cross / norm[:, None]
    centers = vectors.mean(axis=1)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    pairs = np.column_stack([owner[:-1][shared], owner[1:][shared]])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return dict(vectors=vectors, vertices=vertices, faces=faces, centers=centers,
                normals=normals, area=norm / 2, pairs=pairs)


def graph_for_faces(geometry, allowed, max_dihedral_deg=180):
    pairs = geometry['pairs']
    keep = allowed[pairs[:, 0]] & allowed[pairs[:, 1]]
    if max_dihedral_deg < 180:
        dots = np.abs(np.einsum('ij,ij->i', geometry['normals'][pairs[:, 0]],
                               geometry['normals'][pairs[:, 1]]))
        keep &= dots >= np.cos(np.radians(max_dihedral_deg))
    pairs = pairs[keep]
    n = len(allowed)
    lengths = np.linalg.norm(geometry['centers'][pairs[:, 0]] - geometry['centers'][pairs[:, 1]], axis=1)
    return coo_matrix((np.r_[lengths, lengths],
                       (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                      shape=(n, n)).tocsr()


def surface_components(geometry, allowed, min_area=0.12, max_dihedral_deg=45):
    graph = graph_for_faces(geometry, allowed, max_dihedral_deg)
    _, labels = connected_components(graph, directed=False)
    areas = np.bincount(labels, weights=geometry['area'] * allowed)
    return [np.flatnonzero((labels == k) & allowed) for k in np.flatnonzero(areas >= min_area)]


def upper_height_grid(geometry, spacing=0.10):
    """Sample the upper envelope on a physical grid without changing the mesh."""
    points = np.concatenate([geometry['vertices'], geometry['centers']])
    origin = points[:, :2].min(axis=0) - spacing * 2
    shape = np.ceil((points[:, :2].max(axis=0) - origin) / spacing).astype(int) + 3
    xy = np.floor((points[:, :2] - origin) / spacing).astype(int)
    heights = np.full(tuple(shape), -np.inf)
    np.maximum.at(heights, (xy[:, 0], xy[:, 1]), points[:, 2])
    occupied = np.isfinite(heights)
    filled = ndi.binary_fill_holes(ndi.binary_closing(occupied, iterations=2))
    nearest = ndi.distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    heights[~occupied] = heights[tuple(nearest[:, ~occupied])]
    distance = ndi.distance_transform_edt(filled) * spacing
    return {'height': heights, 'footprint': filled, 'edge_distance': distance,
            'origin': origin, 'spacing': spacing}


def sample_grid(grid, points, field):
    xy = np.floor((points[:, :2] - grid['origin']) / grid['spacing']).astype(int)
    xy = np.clip(xy, 0, np.array(grid[field].shape) - 1)
    return grid[field][xy[:, 0], xy[:, 1]]


def crown_reference(geometry, grid):
    """Fit on spatially balanced crown samples, iteratively excluding depressions."""
    ix = np.argwhere(grid['footprint'] & (grid['edge_distance'] >= 0.65))
    xy = grid['origin'] + (ix + 0.5) * grid['spacing']
    z = grid['height'][ix[:, 0], ix[:, 1]]
    points = np.column_stack([xy, z])
    zmin, zmax = geometry['vertices'][:, 2].min(), geometry['vertices'][:, 2].max()
    points = points[points[:, 2] >= zmin + 0.40 * (zmax - zmin)]
    if len(points) < 50:
        raise ValueError('Insufficient crown surface for reference fitting')
    # Centering makes the quadratic fit stable despite large scanner coordinates.
    origin = points[:, :2].mean(axis=0)
    centered = points.copy(); centered[:, :2] -= origin
    coeff = measurement.fit_robust_quadric(centered)
    for _ in range(3):
        residual = measurement.quadric_z(centered[:, 0], centered[:, 1], coeff) - centered[:, 2]
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        keep = residual < med + max(1.5 * mad, 0.20)
        if keep.sum() < 50:
            break
        coeff = measurement.fit_quadric(*centered[keep].T)
    # Convert back to the original coordinate system for compatible measurement.
    a, b, c, d, e, f = coeff; ox, oy = origin
    return np.array([a, b, c, d - 2*a*ox - c*oy, e - 2*b*oy - c*ox,
                     f + a*ox*ox + b*oy*oy + c*ox*oy - d*ox - e*oy])


def discover_candidates(geometry):
    """Search all crown surfaces, including peripheral/deep gingival floors."""
    grid = upper_height_grid(geometry)
    reference = crown_reference(geometry, grid)
    p = geometry['centers']
    depth = measurement.quadric_z(p[:, 0], p[:, 1], reference) - p[:, 2]
    zmin, zmax = geometry['vertices'][:, 2].min(), geometry['vertices'][:, 2].max()
    crown = p[:, 2] >= zmin + 0.12 * (zmax - zmin)
    top_gap = sample_grid(grid, p, 'height') - p[:, 2]
    visible = top_gap < 0.35
    edge = sample_grid(grid, p, 'edge_distance')
    nz = np.abs(geometry['normals'][:, 2])
    smooth_height = ndi.gaussian_filter(grid['height'], 0.7)
    closed = ndi.grey_closing(smooth_height, size=(25, 25))
    grid['concavity'] = np.maximum(closed - smooth_height, 0)
    concavity = sample_grid(grid, p, 'concavity')
    # Contrast thresholds are detection parameters, not clinical target depths.
    allowed = crown & visible & (depth >= 0.50) & (nz >= 0.48)
    components = surface_components(geometry, allowed, min_area=0.12, max_dihedral_deg=42)
    candidates = []
    for ids in components:
        weights = geometry['area'][ids]
        area = float(weights.sum())
        center = np.average(p[ids], axis=0, weights=weights)
        median_depth = measurement.weighted_quantile(depth[ids], weights, 0.5)
        inward = float(np.average(edge[ids], weights=weights))
        concave_fraction = float(np.average(concavity[ids] >= 0.18, weights=weights))
        mean_concavity = float(np.average(concavity[ids], weights=weights))
        if concave_fraction < 0.12 or median_depth < 0.8:
            continue
        score = area ** 0.5 * min(median_depth, 3) * inward ** 2 * (0.3 + concave_fraction)
        candidates.append({'ids': ids, 'area_mm2': area, 'center': center,
                           'depth_median_mm': median_depth, 'edge_distance_mm': inward,
                           'concave_fraction': concave_fraction, 'mean_concavity_mm': mean_concavity,
                           'score': score})
    candidates.sort(key=lambda c: c['score'], reverse=True)
    for i, candidate in enumerate(candidates):
        candidate['id'] = i + 1
    return candidates, reference, {'grid': grid, 'depth': depth, 'crown': crown,
                                   'top_gap': top_gap, 'edge': edge, 'nz': nz, 'concavity': concavity}


def choose_candidates(candidates, geometry):
    if not candidates:
        return None, None
    occlusal = max(candidates, key=lambda c: c['score'])
    options = []
    for c in candidates:
        if c is occlusal or c['edge_distance_mm'] > 2.2 or c['concave_fraction'] < 0.55:
            continue
        gap = float(cKDTree(geometry['centers'][occlusal['ids']]).query(geometry['centers'][c['ids']])[0].min())
        if gap > 4.0 or c['area_mm2'] < 0.15:
            continue
        score = c['area_mm2'] ** 0.65 * min(c['depth_median_mm'], 4) / (0.5 + c['edge_distance_mm'])
        score *= np.exp(-gap / 3)
        options.append((score, c))
    proximal = max(options, key=lambda t: t[0])[1] if options else None
    return occlusal, proximal


def summarize_candidate(ids, geometry, fields, source):
    weights=geometry['area'][ids];p=geometry['centers'][ids]
    inward=float(np.average(fields['edge'][ids],weights=weights))
    return {'ids':np.asarray(ids,int),'area_mm2':float(weights.sum()),
            'center':np.average(p,axis=0,weights=weights),
            'depth_median_mm':measurement.weighted_quantile(fields['depth'][ids],weights,0.5),
            'edge_distance_mm':inward,
            'concave_fraction':float(np.average(fields['concavity'][ids]>=0.18,weights=weights)),
            'source':source}


def partition_opening(candidate, geometry, fields):
    """Separate an open terminal box using width/depth changes along its axis.

    The result is explicitly provisional: an occlusal-only preparation that
    reaches a tooth edge can have similar geometry. No target clinical depth
    or assumption that the box must be deeper is used.
    """
    ids=candidate['ids'];p=geometry['centers'][ids];w=geometry['area'][ids]
    edge=fields['edge'][ids]
    terminal=edge<0.85;interior=edge>2.4
    if w[terminal].sum()<0.25 or w[interior].sum()<0.5:
        return None
    end=np.average(p[terminal,:2],axis=0,weights=w[terminal])
    inner=np.average(p[interior,:2],axis=0,weights=w[interior])
    axis=inner-end;axis/=max(np.linalg.norm(axis),1e-9)
    t=p[:,:2]@axis;t-=np.quantile(t,0.02)
    lateral=p[:,:2]@np.array([-axis[1],axis[0]])
    options=[]
    for cut in np.arange(0.7,min(2.6,np.quantile(t,0.8)),0.15):
        left=(t>=max(0,cut-0.65))&(t<cut)
        right=(t>=cut)&(t<cut+0.65)
        proximal=t<cut;occlusal=~proximal
        if min(left.sum(),right.sum())<8 or min(w[proximal].sum(),w[occlusal].sum())<0.3:
            continue
        if np.average(edge[proximal],weights=w[proximal])>1.65:
            continue
        width_before=np.quantile(lateral[left],0.95)-np.quantile(lateral[left],0.05)
        width_after=np.quantile(lateral[right],0.95)-np.quantile(lateral[right],0.05)
        ratio=width_before/max(width_after,0.10)
        step=abs(measurement.weighted_quantile(p[left,2],w[left],0.5)-
                 measurement.weighted_quantile(p[right,2],w[right],0.5))
        if ratio<1.25 and step<0.22:
            continue
        score=np.log(max(ratio,0.3))+min(step,0.8)-0.25*abs(cut-1.4)
        options.append((score,cut,proximal,ratio,step))
    if not options:
        return None
    score,cut,pselect,ratio,step=max(options,key=lambda v:v[0])
    o=summarize_candidate(ids[~pselect],geometry,fields,'opening_partition_requires_review')
    p=summarize_candidate(ids[pselect],geometry,fields,'opening_partition_requires_review')
    note={'cut_from_opening_mm':float(cut),'width_ratio_across_cut':float(ratio),
          'height_change_across_cut_mm':float(step),'score':float(score)}
    return o,p,note


def select_floors(candidates, geometry, fields, guidance):
    occlusal,proximal=choose_candidates(candidates,geometry)
    notes=[];partition=None
    if occlusal is None:
        return None,None,{'notes':['no_supported_floor_candidates']}
    # Mask scores are available only after registration passes its quality gate.
    for name in ['Occlusal','Proximal']:
        info,support,visible,_=guidance.get(name,({},None,None,None))
        if not info.get('use_for_selection') or support is None:
            continue
        ranking=[]
        for candidate in candidates:
            ids=candidate['ids'];area=geometry['area'][ids]
            coverage=float(np.sum(area*support[ids])/area.sum())
            candidate[name.lower()+'_mask_coverage']=coverage
            if coverage>=0.30:
                ranking.append((coverage*np.sqrt(area.sum()),candidate))
        if ranking:
            chosen=max(ranking,key=lambda v:v[0])[1]
            if name=='Occlusal':occlusal=chosen
            else:proximal=chosen
            notes.append(name.lower()+'_selection_supported_by_registered_mask')
        else:
            notes.append(name.lower()+'_registered_mask_geometry_disagree')
    # One connected floor may contain both the central preparation and its box.
    opening=partition_opening(occlusal,geometry,fields)
    if opening is not None and (proximal is None or proximal is occlusal or
                                 proximal['area_mm2'] < 0.5):
        occlusal,proximal,partition=opening
        notes.append('occlusal_proximal_boundary_estimated_from_opening_requires_review')
    elif proximal is occlusal:
        proximal=None;notes.append('cavity_labels_overlap_unresolved')
    # Recover tiny disconnected floor pieces near the selected central floor.
    if occlusal is not None:
        pieces=[occlusal['ids']]
        tree=cKDTree(geometry['centers'][occlusal['ids']])
        for candidate in candidates:
            if candidate is proximal or np.intersect1d(candidate['ids'],occlusal['ids']).size:
                continue
            if candidate['area_mm2']>1.0 or candidate['concave_fraction']<0.65:
                continue
            gap=float(tree.query(geometry['centers'][candidate['ids']])[0].min())
            if gap<0.35 and abs(candidate['depth_median_mm']-occlusal['depth_median_mm'])<0.6:
                pieces.append(candidate['ids'])
        if len(pieces)>1:
            occlusal=summarize_candidate(np.unique(np.concatenate(pieces)),geometry,fields,'nearby_floor_fragments')
            notes.append('nearby_occlusal_floor_fragments_recovered')
    return occlusal,proximal,{'notes':notes,'opening_partition':partition}


def grow_cavity_regions(geometry, fields, occlusal, proximal):
    """Grow through mesh-connected depressed walls, then resolve competing labels."""
    allowed=fields['crown']&(fields['depth']>0.30)&(fields['top_gap']<0.50)
    graph=graph_for_faces(geometry,allowed)
    distances=[]
    for candidate in [occlusal,proximal]:
        if candidate is None:
            distances.append(np.full(len(allowed),np.inf));continue
        # All floor faces are sources; distance is along actual shared mesh edges.
        distances.append(dijkstra(graph,directed=False,indices=candidate['ids'],
                                   min_only=True,limit=2.5))
    result=[]
    for i in range(2):
        own=distances[i];other=distances[1-i]
        select=np.isfinite(own)&(own<=2.5)&(own<=other if i==0 else own<other)&allowed
        candidate=[occlusal,proximal][i]
        if candidate is not None:select[candidate['ids']]=True
        result.append(np.flatnonzero(select))
    # Floor labels are immutable seeds and always disjoint.
    if occlusal is not None and proximal is not None:
        result[0]=np.setdiff1d(result[0],proximal['ids'])
        result[1]=np.setdiff1d(result[1],occlusal['ids'])
    return result


def case_number(path):
    match = re.search(r'[OP][_-]?(\d+)', Path(path).stem, re.I)
    return int(match.group(1)) if match else None


def matching_file(folder, number):
    matches = [p for p in Path(folder).iterdir() if p.is_file()
               and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
               and case_number(p) == number] if Path(folder).is_dir() else []
    if len(matches) > 1:
        raise ValueError(f'Ambiguous image/mask match for {number} in {folder}')
    return matches[0] if matches else None


def image_tooth_mask(image):
    """Remove white background and disconnected black rotation-padding corners."""
    gray = np.asarray(image.convert('L'))
    foreground = gray < 242
    labels, count = ndi.label(foreground)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    sizes = np.bincount(labels.ravel()); sizes[border] = 0; sizes[0] = 0
    if not sizes.max():
        return None
    tooth = labels == sizes.argmax()
    return ndi.binary_fill_holes(tooth)


def projection_basis(azimuth, elevation, roll):
    az, el, ro = np.radians([azimuth, elevation, roll])
    right = np.array([np.cos(az), np.sin(az), 0.0])
    up = np.array([-np.sin(az)*np.sin(el), np.cos(az)*np.sin(el), np.cos(el)])
    toward_camera = np.cross(right, up)
    basis = np.stack([right, -up])
    rotation = np.array([[np.cos(ro), -np.sin(ro)], [np.sin(ro), np.cos(ro)]])
    return rotation @ basis, toward_camera


def silhouette_from_points(uv, shape):
    pixels = np.rint(uv).astype(int)
    valid = np.isfinite(uv).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < shape[1])
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < shape[0])
    raster = np.zeros(shape, bool)
    raster[pixels[valid, 1], pixels[valid, 0]] = True
    return ndi.binary_fill_holes(ndi.binary_closing(raster, iterations=1))


def project(points, camera):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    matrix = np.asarray(camera['matrix'], float)
    projected = homogeneous @ matrix.T
    if matrix.shape == (2, 4):
        depth = points @ np.asarray(camera['toward_camera'], float)
        return projected, depth
    if matrix.shape == (3, 4):
        positive = projected[:, 2] > 1e-9
        uv = np.full((len(points), 2), -1e6)
        uv[positive] = projected[positive, :2] / projected[positive, 2, None]
        return uv, -projected[:, 2]
    raise ValueError('Camera matrix must be 2x4 orthographic or 3x4 perspective')


def fit_image_camera(vertices, tooth, cavity_type):
    """Estimate an orthographic pose from silhouettes; expose pose ambiguity.

    This is an uncalibrated estimate, not anatomical registration ground truth.
    Independent X/Y scales accommodate preprocessed image resizing, but make
    camera pose less identifiable. Similar competing poses disable mask use.
    """
    size = 96
    target = np.asarray(Image.fromarray(tooth).resize((size, size), Image.Resampling.NEAREST), bool)
    y, x = np.where(target); lower=np.array([x.min(), y.min()]); upper=np.array([x.max(), y.max()])
    source_points = vertices[::max(1, len(vertices)//16000)]
    source_points = source_points - vertices.mean(axis=0)
    target_axis = np.linalg.eigh(np.cov(np.stack([x, y])))[1][:, -1]
    target_angle = np.degrees(np.arctan2(target_axis[1], target_axis[0]))

    def evaluate(angles, return_camera=False):
        basis, direction = projection_basis(*angles)
        uv = source_points @ basis.T
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        scales = (upper - lower) / np.maximum(hi - lo, 1e-5)
        translation = lower - lo * scales
        uv = uv * scales + translation
        render = silhouette_from_points(uv, target.shape)
        iou = np.count_nonzero(render & target) / max(1, np.count_nonzero(render | target))
        if not return_camera:
            return -iou
        pixel_scale = np.array([tooth.shape[1], tooth.shape[0]]) / size
        linear = (scales * pixel_scale)[:, None] * basis
        offset = translation * pixel_scale - linear @ vertices.mean(axis=0)
        return {'matrix': np.column_stack([linear, offset]), 'toward_camera': direction,
                'angles_deg': np.asarray(angles), 'silhouette_iou': iou,
                'image_size': [int(tooth.shape[1]), int(tooth.shape[0])],
                'axis_scale_ratio': float(scales[0] / scales[1])}

    proposals = []
    elevations = [15, 35, 55, 75, 90] if cavity_type == 'Occlusal' else [0, 15, 30, 45]
    for az in range(0, 360, 30):
        for el in elevations:
            basis, _ = projection_basis(az, el, 0)
            uv = source_points @ basis.T
            axis = np.linalg.eigh(np.cov(uv.T))[1][:, -1]
            initial_roll = target_angle - np.degrees(np.arctan2(axis[1], axis[0]))
            for roll in [initial_roll, initial_roll + 180]:
                angles = np.array([az, el, roll])
                proposals.append((evaluate(angles), angles))
    proposals.sort(key=lambda item: item[0])
    refined = []
    for _, initial in proposals[:6]:
        result = minimize(evaluate, initial, method='Powell',
                          bounds=[(initial[0]-25, initial[0]+25),
                                  (max(-10, initial[1]-20), min(90, initial[1]+20)),
                                  (initial[2]-25, initial[2]+25)],
                          options={'maxiter': 12, 'xtol': 0.5, 'ftol': 0.001})
        refined.append(evaluate(result.x, True))
    refined.extend(evaluate(angles, True) for _, angles in proposals[:16])
    refined.sort(key=lambda r: r['silhouette_iou'], reverse=True)
    best = refined[0]
    alternatives=[]
    for option in refined[1:]:
        angle = np.degrees(np.arccos(np.clip(np.dot(best['toward_camera'], option['toward_camera']), -1, 1)))
        linear_difference = np.linalg.norm(np.asarray(best['matrix'])[:, :3] - np.asarray(option['matrix'])[:, :3])
        if angle > 30 or linear_difference > 0.8 * np.linalg.norm(np.asarray(best['matrix'])[:, :3]):
            alternatives.append(option)
    gap = best['silhouette_iou'] - max((o['silhouette_iou'] for o in alternatives), default=0)
    best['competing_pose_iou_gap'] = float(gap)
    best['status'] = ('estimated_usable_requires_review' if best['silhouette_iou'] >= 0.90 and gap >= 0.025
                      else 'ambiguous_pose' if best['silhouette_iou'] >= 0.85 else 'poor_silhouette_fit')
    best['use_for_selection'] = best['status'] == 'estimated_usable_requires_review'
    best['alternatives'] = [{k:v for k,v in a.items() if k != 'alternatives'} for a in alternatives[:3]]
    return best


def mask_face_support(geometry, camera, mask):
    """Project to mask pixels and reject surfaces hidden behind the visible shell."""
    points = np.concatenate([geometry['vertices'], geometry['centers']])
    uv, depth = project(points, camera)
    h,w=mask.shape; pixels=np.rint(uv).astype(int)
    valid=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    zbuffer=np.full((h,w),-np.inf)
    np.maximum.at(zbuffer,(pixels[valid,1],pixels[valid,0]),depth[valid])
    zbuffer=ndi.maximum_filter(zbuffer,size=3)
    uv,depth=project(geometry['centers'],camera); pixels=np.rint(uv).astype(int)
    valid=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    ids=np.flatnonzero(valid)
    visible=np.zeros(len(depth),bool); inside=visible.copy()
    visible[ids]=depth[ids] >= zbuffer[pixels[ids,1],pixels[ids,0]]-0.30
    inside[ids]=mask[pixels[ids,1],pixels[ids,0]]
    return inside & visible, visible


def prepare_mask_guidance(geometry, number, cavity_type, image_folder, mask_folder, cameras=None):
    image_path=matching_file(image_folder,number);mask_path=matching_file(mask_folder,number)
    info={'status':'missing_image_or_mask','use_for_selection':False}
    if image_path is None or mask_path is None:
        return info, None, None, None
    image=Image.open(image_path).convert('RGB');mask_image=Image.open(mask_path).convert('L')
    info.update(image_path=str(image_path.resolve()),mask_path=str(mask_path.resolve()))
    if image.size != mask_image.size:
        info['status']='image_mask_grid_mismatch'
        return info,None,None,None
    mask=np.asarray(mask_image)>0
    if not mask.any():
        info['status']='empty_mask';return info,None,None,None
    tooth=image_tooth_mask(image)
    if tooth is None:
        info['status']='tooth_silhouette_not_found';return info,None,None,None
    given=(cameras or {}).get(str(number),{}).get(cavity_type.lower())
    if given:
        camera=dict(given)
        if camera.get('image_size') != list(image.size):
            info['status']='camera_grid_mismatch';return info,None,None,None
        matrix=np.asarray(camera['matrix'],float)
        if not np.isfinite(matrix).all() or matrix.shape not in [(2,4),(3,4)]:
            raise ValueError('Invalid supplied camera matrix')
        if matrix.shape==(2,4):
            direction=np.asarray(camera['toward_camera'],float)
            if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-10:
                raise ValueError('Invalid orthographic toward_camera vector')
            camera['toward_camera']=direction/np.linalg.norm(direction)
        camera.update(status='supplied_camera_requires_review',use_for_selection=True)
    else:
        camera=fit_image_camera(geometry['vertices'],tooth,cavity_type)
    info.update(camera)
    overlap=float(np.count_nonzero(mask & ndi.binary_dilation(tooth,iterations=2))/mask.sum())
    info['mask_inside_tooth_fraction']=overlap
    if overlap<0.90:
        info.update(status='mask_outside_source_tooth',use_for_selection=False)
    support,visible=mask_face_support(geometry,info,mask)
    return info,support,visible,{'image':image,'mask':mask,'tooth':tooth}


def diagnostic(path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geometry = mesh_geometry(measurement.stlmesh.Mesh.from_file(str(path)).vectors)
    candidates, reference, fields = discover_candidates(geometry)
    occlusal, proximal = choose_candidates(candidates, geometry)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    p=geometry['centers']; step=max(1,len(p)//20000)
    for ax in axes:
        ax.scatter(p[::step,0],p[::step,1],c='0.85',s=1);ax.set_aspect('equal')
    sel=fields['crown'] & (fields['top_gap']<0.35)
    sc=axes[0].scatter(p[sel,0],p[sel,1],c=fields['depth'][sel],s=2,vmin=0,vmax=4,cmap='viridis')
    fig.colorbar(sc,ax=axes[0]);axes[0].set_title('Reference depth, visible crown')
    for c in candidates:
        ids=c['ids'];axes[1].scatter(p[ids,0],p[ids,1],s=3,label=str(c['id']))
        axes[1].text(c['center'][0],c['center'][1],str(c['id']))
    axes[1].set_title('Surface-connected floor candidates')
    for name,c,color in [('O',occlusal,'red'),('P',proximal,'green')]:
        if c:
            ids=c['ids'];axes[2].scatter(p[ids,0],p[ids,1],s=3,c=color,label=name)
    axes[2].legend();axes[2].set_title('Initial selection')
    fig.suptitle(path.stem);fig.savefig(out,dpi=130);plt.close(fig)
    return [{k:measurement.json_safe(v) for k,v in c.items() if k!='ids'} for c in candidates]
