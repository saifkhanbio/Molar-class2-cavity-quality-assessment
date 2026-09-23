"""Image/STL registration experiments; never modifies source inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'landmark_lib'))
import numpy as np
import cv2
from numba import njit
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import json, hashlib, time

BASE = Path(__file__).parent / 'landmark_sources'
CACHE = Path(__file__).parent / 'registration'
CACHE.mkdir(exist_ok=True)

def mesh(path):
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], 'little')
    dt = np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    assert len(data) == 84+50*count
    triangles = np.frombuffer(data,dt,count,84)['v'].astype(float)
    v, inv = np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
    f = inv.reshape(-1,3)
    cross = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn,f[:,k],cross)
    vn /= np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-12)
    return v, f, vn

@njit(cache=True)
def raster(uv, z, normals, faces, size):
    depth = np.full((size,size),-1e30)
    ids = np.full((size,size),-1,np.int32)
    normal = np.zeros((size,size,3))
    for fi in range(len(faces)):
        ia,ib,ic = faces[fi]
        ax,ay=uv[ia]; bx,by=uv[ib]; cx,cy=uv[ic]
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-10:continue
        x0=max(0,int(np.ceil(min(ax,bx,cx))));x1=min(size-1,int(np.floor(max(ax,bx,cx))))
        y0=max(0,int(np.ceil(min(ay,by,cy))));y1=min(size-1,int(np.floor(max(ay,by,cy))))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                a=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den
                b=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den
                c=1-a-b
                if a>=-1e-7 and b>=-1e-7 and c>=-1e-7:
                    zz=a*z[ia]+b*z[ib]+c*z[ic]
                    if zz>depth[y,x]:
                        depth[y,x]=zz;ids[y,x]=fi
                        for k in range(3):normal[y,x,k]=a*normals[ia,k]+b*normals[ib,k]+c*normals[ic,k]
    return depth,ids,normal

def basis(az,el,roll):
    az,el,ro=np.radians([az,el,roll])
    right=np.array([np.cos(az),np.sin(az),0.])
    up=np.array([-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)])
    direct=np.cross(right,up)
    b=np.stack([right,-up])
    rr=np.array([[np.cos(ro),-np.sin(ro)],[np.sin(ro),np.cos(ro)]])
    return rr@b,direct

def image_path(num,view):
    if num<=18:return BASE/('O_M_image' if view=='O' else 'P_M_Image')/f'{view}_{num}.png'
    return BASE/'student_work'/f'{view}_{num}_st.png'

def prepare(path,size=160):
    orig=np.array(Image.open(path).convert('RGB'))
    gray=cv2.cvtColor(orig,cv2.COLOR_RGB2GRAY)
    n,lab,stats,_=cv2.connectedComponentsWithStats((gray<240).astype('uint8'))
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]); mask=lab==idx
    yy,xx=np.where(mask); x0,x1=xx.min(),xx.max();y0,y1=yy.min(),yy.max()
    side=max(x1-x0+1,y1-y0+1)*1.12
    scale=(size-1)/side
    offset=np.array([(size-1)/2-(x0+x1)/2*scale,(size-1)/2-(y0+y1)/2*scale])
    aff=np.column_stack([np.eye(2)*scale,offset])
    crop=cv2.warpAffine(orig,aff,(size,size),borderValue=(255,255,255))
    target=cv2.warpAffine(mask.astype('uint8'),aff,(size,size),flags=cv2.INTER_NEAREST)>0
    yy,xx=np.where(target)
    return dict(orig=orig,crop=crop,gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)/255.,mask=target,
                lo=np.array([xx.min(),yy.min()]),hi=np.array([xx.max(),yy.max()]),
                scale=scale,offset=offset,aff=aff,size=size)

def render(v,f,vn,params,target):
    b,d=basis(*params[:3]); centered=v-v.mean(axis=0)
    uv=centered@b.T; z=centered@d
    perspective=params[7] if len(params)>7 else 0.
    denom=1-perspective*z/np.linalg.norm(np.ptp(v,axis=0))
    uv=uv/denom[:,None]
    lo=uv.min(0);hi=uv.max(0)
    scales=(target['hi']-target['lo'])/(hi-lo)
    off=target['lo']-lo*scales
    if len(params)>3:
        scales*=np.exp(params[3:5]);off=target['lo']-lo*scales+params[5:7]
    uv=uv*scales+off
    nn=np.column_stack([vn@b.T,vn@d])
    depth,ids,norm=raster(uv,z,nn,f,target['size'])
    denominator=np.r_[-perspective*d/np.linalg.norm(np.ptp(v,axis=0)),
                      1+perspective*(v.mean(0)@d)/np.linalg.norm(np.ptp(v,axis=0))]
    matrix=np.vstack([np.column_stack([scales[:,None]*b,-scales*(v.mean(0)@b.T)])+off[:,None]*denominator,denominator])
    return depth,ids,norm,matrix,d

def scores(rendered,t,detail=False):
    dep,ids,norm,*_=rendered
    mask=ids>=0; tgt=t['mask']; inter=mask&tgt
    iou=inter.sum()/max(1,(mask|tgt).sum())
    use=ndi.binary_erosion(inter,iterations=2)
    # Learn view-space diffuse lighting rather than assume the screenshot light direction.
    y=t['gray'][use]
    if len(y)<100:return (10.,0.,0.) if detail else 10.
    n=norm[use];n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    design=np.column_stack([np.ones(len(n)),n,n[:,2]**2])
    weights=np.linalg.lstsq(design,y,rcond=None)[0]
    pred=design@weights
    ncc=np.corrcoef(pred,y)[0,1]
    if not np.isfinite(ncc): ncc=0
    error=1-iou+0.32*(1-ncc)
    return (float(error),float(iou),float(ncc)) if detail else error

def fit(num,view,quick=False):
    start=time.time();v,f,vn=mesh(BASE/'STLFILES'/f'O_{num}.stl')
    t=prepare(image_path(num,view),112)
    proposals=[]
    elevations=[60,75,90] if view=='O' else [-10,10,25]
    for az in range(0,360,30):
        for el in elevations:
            for ro in (0,90,180,270):
                p=np.array([az,el,ro],float)
                loss=scores(render(v,f,vn,p,t),t)
                proposals.append((loss,p))
    proposals.sort(key=lambda a:a[0]); results=[]
    for loss,p in proposals[:(3 if quick else 5)]:
        result=minimize(lambda a:scores(render(v,f,vn,a,t),t),p,method='Powell',
                        bounds=[(p[0]-30,p[0]+30),(max(-50,p[1]-25),min(110,p[1]+25)),(p[2]-45,p[2]+45)],
                        options={'maxiter':15,'xtol':.05,'ftol':.0001})
        results.append((float(result.fun),result.x))
    results.sort(key=lambda a:a[0]);p=np.r_[results[0][1],[0,0,0,0,0.1]]
    t=prepare(image_path(num,view),256)
    result=minimize(lambda a:scores(render(v,f,vn,a,t),t),p,method='Powell',
                    bounds=[(p[0]-8,p[0]+8),(p[1]-8,p[1]+8),(p[2]-8,p[2]+8),(-.08,.08),(-.08,.08),(-6,6),(-6,6),(0.,0.85)],
                    options={'maxiter':12,'xtol':.02,'ftol':.00005})
    p=result.x; rr=render(v,f,vn,p,t);loss,iou,ncc=scores(rr,t,True)
    camera=rr[3].copy();camera[:2]=(camera[:2]-t['offset'][:,None]*camera[2])/t['scale']
    report=dict(case=num,view=view,image=str(image_path(num,view).relative_to(BASE)),parameters=p.tolist(),
                matrix=camera.tolist(),toward_camera=rr[4].tolist(),silhouette_iou=iou,shading_correlation=ncc,
                objective=loss,alternatives=[{'objective':a,'angles':b.tolist()} for a,b in results],
                image_size=[t['orig'].shape[1],t['orig'].shape[0]],seconds=time.time()-start)
    (CACHE/f'{view}_{num}.json').write_text(json.dumps(report,indent=2))
    preview(v,f,vn,p,t,num,view)
    print(json.dumps({k:report[k] for k in ['case','view','silhouette_iou','shading_correlation','seconds']}),flush=True)
    return report

def preview(v,f,vn,p,t,num,view):
    dep,ids,norm,*_=render(v,f,vn,p,t);mask=ids>=0
    norm/=np.maximum(np.linalg.norm(norm,axis=2,keepdims=True),1e-12)
    shade=np.clip(0.2+0.75*np.abs(norm[:,:,2]),0,1)
    gray=np.uint8(shade*255);gray[~mask]=255
    panel=np.repeat(gray[:,:,None],3,axis=2)
    over=t['crop'].copy(); contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over,contours,-1,(255,30,30),1)
    out=np.concatenate([t['crop'],panel,over],axis=1)
    Image.fromarray(out).save(CACHE/f'{view}_{num}_fit.png')

def refine_existing(num,view):
    path=CACHE/f'{view}_{num}.json'; report=json.loads(path.read_text())
    p=np.array(report['parameters']);p=np.r_[p,0.1] if len(p)==7 else p
    v,f,vn=mesh(BASE/'STLFILES'/f'O_{num}.stl');t=prepare(image_path(num,view),256)
    initial=scores(render(v,f,vn,p,t),t)
    result=minimize(lambda a:scores(render(v,f,vn,a,t),t),p,method='Powell',
        bounds=[(p[0]-10,p[0]+10),(p[1]-10,p[1]+10),(p[2]-10,p[2]+10),(-.10,.10),(-.10,.10),(-8,8),(-8,8),(0.,1.5)],
        options={'maxiter':18,'xtol':.008,'ftol':.00001})
    if result.fun<initial:p=result.x
    rr=render(v,f,vn,p,t);loss,iou,ncc=scores(rr,t,True)
    camera=rr[3].copy();camera[:2]=(camera[:2]-t['offset'][:,None]*camera[2])/t['scale']
    report.update(parameters=p.tolist(),matrix=camera.tolist(),toward_camera=rr[4].tolist(),
                  silhouette_iou=iou,shading_correlation=ncc,objective=loss,refined=True)
    path.write_text(json.dumps(report,indent=2));preview(v,f,vn,p,t,num,view)
    print('REFINED',view,num,round(iou,5),round(ncc,5),'perspective',round(p[7],4),flush=True)
    return report

if __name__=='__main__':
    for arg in sys.argv[1:] or ['O_1','P_1']:
        view,num=arg.split('_');fit(int(num),view)
