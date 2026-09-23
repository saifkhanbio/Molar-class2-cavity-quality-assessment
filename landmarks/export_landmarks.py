"""Export reviewable image/STL candidate landmark correspondences."""
import csv, json, hashlib, shutil, html, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
import derive_landmarks as d

OUT=Path(__file__).parent.parent/'outputs'/'image_stl_landmarks'
ROOT=str(d.BASE.resolve())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def cavity_mask(num,view,orig):
    if num<=18:
        path=d.BASE/f'{view}_M_mask'/f'{view}_{num}.png'
        mask=cv2.imread(str(path),0)
        assert mask.shape==orig.shape[:2]
        return mask>127, dict(mask=str(path.relative_to(d.BASE)),mask_kind='existing_reference_mask',mask_image_ecc=1.0,mask_to_original_matrix=np.eye(2,3).tolist())
    path=d.BASE/f'pred_{view}_M_masks_folder'/f'{view}_{num}_st_mask.png'
    mask=cv2.imread(str(path),0)
    sample_path=d.BASE/f'samples_stu_{view}'/f'{view}_{num}_st.png'
    sample=cv2.imread(str(sample_path),0)
    target=cv2.resize(sample,(256,256)).astype('float32')/255
    raw=cv2.resize(cv2.cvtColor(orig,cv2.COLOR_RGB2GRAY),(256,256)).astype('float32')/255
    cc,warp=cv2.findTransformECC(target,raw,np.eye(2,3,dtype='float32'),cv2.MOTION_AFFINE,
                  (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6))
    assert cc>.97, (num,view,'Mask image correspondence failed',cc)
    height,width=orig.shape[:2];mh,mw=mask.shape
    to_small=np.array([[256/mw,0,128/mw-.5],[0,256/mh,128/mh-.5],[0,0,1]])
    to_raw=np.array([[width/256,0,width/512-.5],[0,height/256,height/512-.5],[0,0,1]])
    matrix=(to_raw@np.vstack([warp,[0,0,1]])@to_small)[:2]
    mapped=cv2.warpAffine(mask,matrix,(width,height),flags=cv2.INTER_NEAREST)>127
    return mapped,dict(mask=str(path.relative_to(d.BASE)),mask_kind='existing_predicted_mask',
         mask_image_ecc=float(cc),mask_to_original_matrix=matrix.tolist(),sample_image=str(sample_path.relative_to(d.BASE)))

def contour(mask):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return max(contours,key=cv2.contourArea)

def image_landmarks(orig,cavity):
    gray=cv2.cvtColor(orig,cv2.COLOR_RGB2GRAY)
    n,lab,stats,_=cv2.connectedComponentsWithStats((gray<240).astype('uint8'))
    outer=contour(lab==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).reshape(-1,2).astype(float)
    directions=[('top',0,-1),('top_right',1,-1),('right',1,0),('bottom_right',1,1),
                ('bottom',0,1),('bottom_left',-1,1),('left',-1,0),('top_left',-1,-1)]
    result=[]
    for name,x,y in directions:
        score=outer@np.array([x,y]);best=outer[np.argmax(score)]
        if any(np.linalg.norm(best-p['uv'])<6 for p in result):continue
        result.append(dict(id=f'T{len(result)+1}',kind='tooth_silhouette_extremum',label=f'tooth_outline_{name}',uv=best))
    if cavity.any():
        cnt=contour(cavity);per=cv2.arcLength(cnt,True)
        poly=cv2.approxPolyDP(cnt,.013*per,True).reshape(-1,2).astype(float)
        if len(poly)>10:
            before=poly-np.roll(poly,1,axis=0);after=np.roll(poly,-1,axis=0)-poly
            bend=1-(before*after).sum(1)/np.maximum(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1),1e-9)
            keep=np.argsort(bend)[-10:];poly=poly[np.sort(keep)]
        # IDs follow boundary order, starting at the top-most corner; no anatomical identity implied.
        first=np.lexsort((poly[:,0],poly[:,1]))[0];poly=np.roll(poly,-first,axis=0)
        for i,point in enumerate(poly):result.append(dict(id=f'C{i+1}',kind='cavity_mask_boundary_corner',label=f'cavity_boundary_corner_{i+1:02d}',uv=point))
    return result

class SurfaceProjector:
    def __init__(self,v,f,vn,camera,toward,orig_path):
        self.v=v;self.f=f;self.tri=v[f];self.camera=np.array(camera);self.toward=np.array(toward)
        projected=np.column_stack([v,np.ones(len(v))])@self.camera.T
        self.q=projected[:,2];self.uv=projected[:,:2]/self.q[:,None]
        screen=self.uv[f]
        self.a=screen[:,0];self.b=screen[:,1];self.c=screen[:,2]
        self.den=(self.b[:,1]-self.c[:,1])*(self.a[:,0]-self.c[:,0])+(self.c[:,0]-self.b[:,0])*(self.a[:,1]-self.c[:,1])
        self.t=d.prepare(orig_path,512)
        small=self.uv*self.t['scale']+self.t['offset']
        z=v@self.toward
        self.depth,self.ids,self.norm=d.raster(small,z,vn,f,512)
        self.nearest=ndi.distance_transform_edt(self.ids<0,return_distances=False,return_indices=True)

    def hit(self,point):
        x,y=point
        with np.errstate(divide='ignore',invalid='ignore'):
            a=((self.b[:,1]-self.c[:,1])*(x-self.c[:,0])+(self.c[:,0]-self.b[:,0])*(y-self.c[:,1]))/self.den
            b=((self.c[:,1]-self.a[:,1])*(x-self.c[:,0])+(self.a[:,0]-self.c[:,0])*(y-self.c[:,1]))/self.den
        c=1-a-b
        fi=np.flatnonzero((a>=-1e-8)&(b>=-1e-8)&(c>=-1e-8)&np.isfinite(a+b+c))
        if not len(fi):return None
        bary=np.column_stack([a[fi],b[fi],c[fi]])/self.q[self.f[fi]]
        bary/=bary.sum(1,keepdims=True)
        xyz=(self.tri[fi]*bary[:,:,None]).sum(1)
        j=np.argmax(xyz@self.toward)
        return int(fi[j]),bary[j],xyz[j]

    def lift(self,point):
        hit=self.hit(point);ray=point.copy()
        if hit is None:
            pixel=np.rint(point*self.t['scale']+self.t['offset']).astype(int).clip(0,511)
            iy,ix=self.nearest[:,pixel[1],pixel[0]]
            ray=(np.array([ix,iy])-self.t['offset'])/self.t['scale'];hit=self.hit(ray)
        return hit,ray

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def picked_points(path,mesh_name,rows):
    root=ET.Element('PickedPoints');info=ET.SubElement(root,'DocumentData')
    ET.SubElement(info,'DataFileName',name=mesh_name)
    ET.SubElement(info,'templateName',name='Estimated image landmarks - review required')
    for row in rows:
        if row['stl_x']=='':continue
        ET.SubElement(root,'point',name=row['landmark_id']+'_'+row['landmark_label'],active='1',
                      x=str(row['stl_x']),y=str(row['stl_y']),z=str(row['stl_z']))
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def draw_preview(projector,rows,camera,num,view,mask):
    t=projector.t;crop=t['crop'].copy();norm=projector.norm.copy()
    b,direct=d.basis(*camera['parameters'][:3]);view_normal=norm@direct
    view_normal/=np.maximum(np.linalg.norm(norm,axis=2),1e-12)
    shade=np.uint8(np.clip(.2+.75*np.abs(view_normal),0,1)*255);shade[projector.ids<0]=255
    model=np.repeat(shade[:,:,None],3,axis=2)
    resized=cv2.warpAffine(mask.astype('uint8'),t['aff'],(512,512),flags=cv2.INTER_NEAREST)
    contours,_=cv2.findContours(resized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop,contours,-1,(221,154,18),1)
    panels=[Image.fromarray(crop),Image.fromarray(model)]
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    except OSError:font=ImageFont.load_default()
    for index,panel in enumerate(panels):
        painter=ImageDraw.Draw(panel);occupied=[]
        for row in rows:
            if index==1 and row['stl_x']=='':continue
            uv=np.array([row['image_x_px'],row['image_y_px']]) if index==0 else np.array([row['surface_projection_x_px'],row['surface_projection_y_px']])
            x,y=uv*t['scale']+t['offset'];color='#007a96' if row['landmark_id'].startswith('T') else '#b06400'
            painter.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='white')
            label=row['landmark_id'];bbox=font.getbbox(label);w=bbox[2]-bbox[0]+6;h=19
            for dx,dy in [(7,-10),(-w-7,-10),(7,8),(-w-7,8),(3,-28),(3,22)]:
                xx=min(510-w,max(2,x+dx));yy=min(490,max(2,y+dy));box=(xx,yy,xx+w,yy+h)
                if not any(box[0]<r[2] and box[2]>r[0] and box[1]<r[3] and box[3]>r[1] for r in occupied):break
            occupied.append(box);painter.rectangle(box,fill='white');painter.text((box[0]+3,box[1]),label,font=font,fill=color)
    out=Image.new('RGB',(1056,586),'#f4f5f7');out.paste(panels[0],(8,42));out.paste(panels[1],(536,42))
    painter=ImageDraw.Draw(out);painter.text((16,12),f'{view}_{num}  |  Image landmarks',font=font,fill='#13233c')
    painter.text((544,12),'Matched locations on the projected STL',font=font,fill='#13233c')
    painter.text((16,562),'T: tooth outline   C: cavity mask corners   |   Estimated correspondences; review required',font=font,fill='#13233c')
    out.save(OUT/'previews'/f'{view}_{num}.png')

def main(numbers=None):
    for sub in ['previews','picked_points','per_view','scripts']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    all_rows=[];summaries=[];cameras={};audit=[];views=[]
    for num in (numbers or list(range(1,19))+list(range(35,46))):
        v,f,vn=d.mesh(d.BASE/'STLFILES'/f'O_{num}.stl')
        for view in ['O','P']:
            cache_num=num-10 if 11<=num<=18 else num
            camera=json.loads((d.CACHE/f'{view}_{cache_num}.json').read_text())
            path=d.image_path(num,view);orig=np.array(Image.open(path).convert('RGB'))
            if cache_num!=num:
                assert sha(path)==sha(d.image_path(cache_num,view))
                assert sha(d.BASE/'STLFILES'/f'O_{num}.stl')==sha(d.BASE/'STLFILES'/f'O_{cache_num}.stl')
            mask,mask_info=cavity_mask(num,view,orig)
            landmarks=image_landmarks(orig,mask)
            projector=SurfaceProjector(v,f,vn,camera['matrix'],camera['toward_camera'],path)
            rows=[]
            for point in landmarks:
                hit,ray=projector.lift(point['uv']);snap=float(np.linalg.norm(ray-point['uv']))
                valid=hit is not None and snap<=6
                if valid:
                    fi,bary,xyz=hit
                    uvh=np.r_[xyz,1]@np.array(camera['matrix']).T
                    check=float(np.linalg.norm(uvh[:2]/uvh[2]-ray))
                    assert check<1e-6
                    assert np.linalg.norm(bary@v[f[fi]]-xyz)<1e-8
                    status='surface_estimate_requires_review' if snap<=2 else 'boundary_snap_requires_review'
                else:
                    fi='';bary=['']*3;xyz=['']*3;check='';status='unresolved_no_surface_within_6px'
                row=dict(mesh_file=f'STLFILES/O_{num}.stl',case_id=num,view='occlusal' if view=='O' else 'proximal',
                    image_file=path.relative_to(d.BASE).as_posix(),image_width_px=orig.shape[1],image_height_px=orig.shape[0],
                    landmark_id=point['id'],landmark_type=point['kind'],landmark_label=point['label'],
                    image_x_px=float(point['uv'][0]),image_y_px=float(point['uv'][1]),
                    stl_x=float(xyz[0]) if valid else '',stl_y=float(xyz[1]) if valid else '',stl_z=float(xyz[2]) if valid else '',
                    face_index_zero_based=fi,barycentric_0=float(bary[0]) if valid else '',
                    barycentric_1=float(bary[1]) if valid else '',barycentric_2=float(bary[2]) if valid else '',
                    surface_projection_x_px=float(ray[0]) if valid else '',surface_projection_y_px=float(ray[1]) if valid else '',
                    boundary_snap_px=snap,projection_consistency_px=check,
                    silhouette_iou=camera['silhouette_iou'],shading_correlation=camera['shading_correlation'],
                    mask_source=mask_info['mask'].replace('\\','/'),mask_kind=mask_info['mask_kind'],status=status,
                    human_verified=False,native_stl_units='unspecified_project_assumes_mm')
                rows.append(row)
            all_rows.extend(rows);write_csv(OUT/'per_view'/f'{view}_{num}.csv',rows)
            picked_points(OUT/'picked_points'/f'{view}_{num}.pp',f'O_{num}.stl',rows)
            draw_preview(projector,rows,camera,num,view,mask)
            nvalid=sum(r['stl_x']!='' for r in rows)
            summary=dict(case_id=num,view='occlusal' if view=='O' else 'proximal',image_file=path.relative_to(d.BASE).as_posix(),
                 landmarks=len(rows),surface_mapped=nvalid,unresolved=len(rows)-nvalid,
                 silhouette_iou=camera['silhouette_iou'],shading_correlation=camera['shading_correlation'],
                 max_boundary_snap_px=max(r['boundary_snap_px'] for r in rows),
                 mask_image_ecc=mask_info['mask_image_ecc'],same_mesh_and_image_as=cache_num if cache_num!=num else '',
                 status='strong_render_fit_requires_landmark_review' if camera['silhouette_iou']>=.97 and camera['shading_correlation']>=.94 else 'registration_requires_review')
            summaries.append(summary)
            cameras.setdefault(str(num),{})[summary['view']]={
                'matrix':camera['matrix'],'toward_camera':camera['toward_camera'],'image_size':[orig.shape[1],orig.shape[0]],
                'source_image':path.relative_to(d.BASE).as_posix(),'projection_model':'estimated_perspective_with_independent_xy_scales',
                'silhouette_iou':camera['silhouette_iou'],'shading_correlation':camera['shading_correlation'],'human_verified':False}
            audit.append(dict(case_id=num,view=view,mesh_sha256=sha(d.BASE/'STLFILES'/f'O_{num}.stl'),image_sha256=sha(path),
                              registration=camera,mask_mapping=mask_info))
            views.append(dict(key=f'{view}_{num}',summary=summary,rows=rows))
            print('EXPORTED',view,num,len(rows),nvalid,flush=True)
    write_csv(OUT/'landmarks.csv',all_rows);write_csv(OUT/'registration_summary.csv',summaries)
    (OUT/'camera_params.json').write_text(json.dumps(cameras,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(dict(source_project=ROOT,pixel_convention='Zero-based pixel centers, x right, y down',records=audit),indent=2))
    build_report(views,all_rows,summaries)
    return all_rows,summaries

def build_report(views,rows,summaries):
    cases=set(s['case_id'] for s in summaries)
    distinct_cases=set(n-10 if 11<=n<=18 else n for n in cases)
    data=json.dumps(views).replace('</','<\/')
    document='''<!doctype html><html><head><meta charset="utf-8"><title>Image to STL landmarks</title>
<style>body{margin:0;background:#edf1f5;color:#14253e;font:15px/1.5 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:32px}header{display:flex;justify-content:space-between;gap:25px}h1{font-size:32px;letter-spacing:-1px;margin:0}p{max-width:900px}.badge{font-size:12px;font-weight:700;color:#8d5600;background:#fff1d6;padding:7px 12px;border-radius:20px;white-space:nowrap}a{color:#00667d}nav{display:flex;gap:18px;margin:20px 0}.card{background:white;padding:22px;border:1px solid #d6dfe9;border-radius:12px;margin:18px 0}select{font:inherit;padding:8px 15px;min-width:180px;border:1px solid #bac9d8;border-radius:7px}.metrics{display:flex;gap:30px;flex-wrap:wrap;margin:16px 0}.metrics b{display:block;font-size:25px}.metrics span{font-size:12px;color:#52657a}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;text-align:left;border-bottom:1px solid #e0e7ee}th{background:#f3f6f9}.muted{color:#56677a}.scroll{overflow:auto}summary{cursor:pointer;font-weight:650}footer{font-size:12px;color:#52657a;margin-top:28px}.note{border-left:4px solid #d69a24;padding:8px 16px;background:#fff8e9}</style></head>
<body><main><header><div><h1>Image → STL landmarks</h1><p class="muted">29 STL filenames · 21 distinct meshes · 58 image views</p></div><div><span class="badge">ESTIMATED · REVIEW REQUIRED</span></div></header>
<p>Candidate correspondences for visible tooth outlines and cavity-mask corners. Each mapped 3D point lies on an original STL triangle. Both occlusal and proximal views are included.</p>
<nav><a href="landmarks.csv">All coordinates (CSV)</a><a href="camera_params.json">Camera matrices</a><a href="registration_summary.csv">Fit summary</a><a href="README.md">Method and limitations</a></nav>
<div class="note">These are model-based surface estimates, not independently annotated anatomical ground truth. Outline overlap and shading correlation assess the rendering fit; they do not measure landmark localization accuracy. T labels are view-dependent outline extrema. C labels are corners of existing cavity masks, not verified anatomical names.</div>
<section class="card"><label for="view">Choose a mesh and view &nbsp;</label><select id="view"></select><div class="metrics" id="metrics"></div><p id="source" class="muted"></p><img id="preview" alt="Numbered image landmarks and matching locations on the projected STL"><p id="links"></p></section>
<section class="card"><h2>Landmark coordinates</h2><p class="muted">Pixels use the original image grid: origin at top left, x right, y down. XYZ uses the untouched STL coordinate frame. STL units are unspecified; the project assumes millimeters.</p><div class="scroll"><table><thead><tr><th>ID</th><th>Type</th><th>Image x</th><th>Image y</th><th>STL x</th><th>STL y</th><th>STL z</th><th>Snap (px)</th><th>Status</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="card"><details><summary>How to use these results</summary><p>Open the matching O_n.stl in MeshLab and load the view's .pp file with the PickPoints tool. Use the numbered preview to review or adjust each point. The CSV also records the original triangle index and barycentric coordinates, allowing exact reconstruction of every surface point.</p><p>The two views have separate landmark sets. IDs such as C1 are local to a view; equal labels across views or teeth do not imply anatomical identity. Do not use these projected points as independent evidence to validate the camera that produced them.</p><p>Files O_11 through O_18 duplicate O_1 through O_8, respectively. Corresponding source images are also identical; results were reused only after checking both file hashes.</p></details></section><footer>Original input files were preserved. No human verification or absolute scale calibration is claimed.</footer>
<script>const data=DATA;const sel=document.querySelector('#view');for(const v of data){const o=document.createElement('option');o.value=v.key;o.textContent='O_'+v.summary.case_id+' · '+v.summary.view;sel.append(o)}const fmt=x=>x===''?'—':Number(x).toFixed(3);function show(){const v=data.find(x=>x.key===sel.value),s=v.summary;document.querySelector('#metrics').innerHTML=`<div><b>${(100*s.silhouette_iou).toFixed(1)}%</b><span>Outline overlap (IoU)</span></div><div><b>${s.shading_correlation.toFixed(3)}</b><span>Shading correlation</span></div><div><b>${s.surface_mapped}/${s.landmarks}</b><span>Points mapped to surface</span></div>`;document.querySelector('#source').textContent=s.image_file+' · '+s.status.replaceAll('_',' ');document.querySelector('#preview').src='previews/'+v.key+'.png';document.querySelector('#links').innerHTML=`<a href="per_view/${v.key}.csv">This view's CSV</a> &nbsp;·&nbsp; <a href="picked_points/${v.key}.pp">MeshLab picked points</a> &nbsp;·&nbsp; <a href="previews/${v.key}.png">Full preview</a>`;document.querySelector('#rows').innerHTML=v.rows.map(r=>`<tr><td><b>${r.landmark_id}</b></td><td>${r.landmark_type.replaceAll('_',' ')}</td><td>${fmt(r.image_x_px)}</td><td>${fmt(r.image_y_px)}</td><td>${fmt(r.stl_x)}</td><td>${fmt(r.stl_y)}</td><td>${fmt(r.stl_z)}</td><td>${fmt(r.boundary_snap_px)}</td><td>${r.status.replaceAll('_',' ')}</td></tr>`).join('')}sel.onchange=show;show();</script></main></body></html>'''.replace('DATA',data)
    document=document.replace('29 STL filenames · 21 distinct meshes · 58 image views',f'{len(cases)} STL filenames · {len(distinct_cases)} distinct meshes · {len(summaries)} image views')
    (OUT/'review.html').write_text(document,encoding='utf-8')
    stats=dict(mesh_files=len(cases),distinct_meshes=len(distinct_cases),views=len(summaries),landmarks=len(rows),surface_mapped=sum(r['stl_x']!='' for r in rows),
       unresolved=sum(r['stl_x']=='' for r in rows),minimum_iou=min(s['silhouette_iou'] for s in summaries),
       median_iou=float(np.median([s['silhouette_iou'] for s in summaries])),minimum_shading_correlation=min(s['shading_correlation'] for s in summaries),
       maximum_projection_consistency_px=max(r['projection_consistency_px'] for r in rows if r['projection_consistency_px']!=''),
       boundary_snap_over_2px=sum(r['boundary_snap_px']>2 for r in rows))
    (OUT/'validation.json').write_text(json.dumps(stats,indent=2));print('TOTAL',json.dumps(stats),flush=True)

if __name__=='__main__':main()
